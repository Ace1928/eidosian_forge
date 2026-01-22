from .compat import escape
from .jsonify import encode
class MakoRenderer(object):
    """
        Defines the builtin ``Mako`` renderer.
        """

    def __init__(self, path, extra_vars):
        self.loader = TemplateLookup(directories=[path], output_encoding='utf-8')
        self.extra_vars = extra_vars

    def render(self, template_path, namespace):
        """
            Implements ``Mako`` rendering.
            """
        tmpl = self.loader.get_template(template_path)
        return tmpl.render(**self.extra_vars.make_ns(namespace))