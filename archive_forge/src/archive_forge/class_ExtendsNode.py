import posixpath
from collections import defaultdict
from django.utils.safestring import mark_safe
from .base import Node, Template, TemplateSyntaxError, TextNode, Variable, token_kwargs
from .library import Library
class ExtendsNode(Node):
    must_be_first = True
    context_key = 'extends_context'

    def __init__(self, nodelist, parent_name, template_dirs=None):
        self.nodelist = nodelist
        self.parent_name = parent_name
        self.template_dirs = template_dirs
        self.blocks = {n.name: n for n in nodelist.get_nodes_by_type(BlockNode)}

    def __repr__(self):
        return '<%s: extends %s>' % (self.__class__.__name__, self.parent_name.token)

    def find_template(self, template_name, context):
        """
        This is a wrapper around engine.find_template(). A history is kept in
        the render_context attribute between successive extends calls and
        passed as the skip argument. This enables extends to work recursively
        without extending the same template twice.
        """
        history = context.render_context.setdefault(self.context_key, [self.origin])
        template, origin = context.template.engine.find_template(template_name, skip=history)
        history.append(origin)
        return template

    def get_parent(self, context):
        parent = self.parent_name.resolve(context)
        if not parent:
            error_msg = "Invalid template name in 'extends' tag: %r." % parent
            if self.parent_name.filters or isinstance(self.parent_name.var, Variable):
                error_msg += " Got this from the '%s' variable." % self.parent_name.token
            raise TemplateSyntaxError(error_msg)
        if isinstance(parent, Template):
            return parent
        if isinstance(getattr(parent, 'template', None), Template):
            return parent.template
        return self.find_template(parent, context)

    def render(self, context):
        compiled_parent = self.get_parent(context)
        if BLOCK_CONTEXT_KEY not in context.render_context:
            context.render_context[BLOCK_CONTEXT_KEY] = BlockContext()
        block_context = context.render_context[BLOCK_CONTEXT_KEY]
        block_context.add_blocks(self.blocks)
        for node in compiled_parent.nodelist:
            if not isinstance(node, TextNode):
                if not isinstance(node, ExtendsNode):
                    blocks = {n.name: n for n in compiled_parent.nodelist.get_nodes_by_type(BlockNode)}
                    block_context.add_blocks(blocks)
                break
        with context.render_context.push_state(compiled_parent, isolated_context=False):
            return compiled_parent._render(context)