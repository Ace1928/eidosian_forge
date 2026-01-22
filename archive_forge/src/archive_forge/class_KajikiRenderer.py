from .compat import escape
from .jsonify import encode
class KajikiRenderer(object):
    """
        Defines the builtin ``Kajiki`` renderer.
        """

    def __init__(self, path, extra_vars):
        self.loader = FileLoader(path, reload=True)
        self.extra_vars = extra_vars

    def render(self, template_path, namespace):
        """
            Implements ``Kajiki`` rendering.
            """
        Template = self.loader.import_(template_path)
        stream = Template(self.extra_vars.make_ns(namespace))
        return stream.render()