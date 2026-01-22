from .compat import escape
from .jsonify import encode
class RendererFactory(object):
    """
    Manufactures known Renderer objects.

    :param custom_renderers: custom-defined renderers to manufacture
    :param extra_vars: extra vars for the template namespace
    """

    def __init__(self, custom_renderers={}, extra_vars={}):
        self._renderers = {}
        self._renderer_classes = dict(_builtin_renderers)
        self.add_renderers(custom_renderers)
        self.extra_vars = ExtraNamespace(extra_vars)

    def add_renderers(self, custom_dict):
        """
        Adds a custom renderer.

        :param custom_dict: a dictionary of custom renderers to add
        """
        self._renderer_classes.update(custom_dict)

    def available(self, name):
        """
        Returns true if queried renderer class is available.

        :param name: renderer name
        """
        return name in self._renderer_classes

    def get(self, name, template_path):
        """
        Returns the renderer object.

        :param name: name of the requested renderer
        :param template_path: path to the template
        """
        if name not in self._renderers:
            cls = self._renderer_classes.get(name)
            if cls is None:
                return None
            else:
                self._renderers[name] = cls(template_path, self.extra_vars)
        return self._renderers[name]

    def keys(self, *args, **kwargs):
        return self._renderer_classes.keys(*args, **kwargs)