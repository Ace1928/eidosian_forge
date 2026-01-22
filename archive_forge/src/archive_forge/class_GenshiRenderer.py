from .compat import escape
from .jsonify import encode
class GenshiRenderer(object):
    """
        Defines the builtin ``Genshi`` renderer.
        """

    def __init__(self, path, extra_vars):
        self.loader = TemplateLoader([path], auto_reload=True)
        self.extra_vars = extra_vars

    def render(self, template_path, namespace):
        """
            Implements ``Genshi`` rendering.
            """
        tmpl = self.loader.load(template_path)
        stream = tmpl.generate(**self.extra_vars.make_ns(namespace))
        return stream.render('html')