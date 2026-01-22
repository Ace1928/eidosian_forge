import posixpath
from collections import defaultdict
from django.utils.safestring import mark_safe
from .base import Node, Template, TemplateSyntaxError, TextNode, Variable, token_kwargs
from .library import Library
class IncludeNode(Node):
    context_key = '__include_context'

    def __init__(self, template, *args, extra_context=None, isolated_context=False, **kwargs):
        self.template = template
        self.extra_context = extra_context or {}
        self.isolated_context = isolated_context
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f'<{self.__class__.__qualname__}: template={self.template!r}>'

    def render(self, context):
        """
        Render the specified template and context. Cache the template object
        in render_context to avoid reparsing and loading when used in a for
        loop.
        """
        template = self.template.resolve(context)
        if not callable(getattr(template, 'render', None)):
            template_name = template or ()
            if isinstance(template_name, str):
                template_name = (construct_relative_path(self.origin.template_name, template_name),)
            else:
                template_name = tuple(template_name)
            cache = context.render_context.dicts[0].setdefault(self, {})
            template = cache.get(template_name)
            if template is None:
                template = context.template.engine.select_template(template_name)
                cache[template_name] = template
        elif hasattr(template, 'template'):
            template = template.template
        values = {name: var.resolve(context) for name, var in self.extra_context.items()}
        if self.isolated_context:
            return template.render(context.new(values))
        with context.push(**values):
            return template.render(context)