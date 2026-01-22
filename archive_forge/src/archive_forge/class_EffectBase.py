from kivy.clock import Clock
from kivy.uix.relativelayout import RelativeLayout
from kivy.properties import (StringProperty, ObjectProperty, ListProperty,
from kivy.graphics import (RenderContext, Fbo, Color, Rectangle,
from kivy.event import EventDispatcher
from kivy.base import EventLoop
from kivy.resources import resource_find
from kivy.logger import Logger
class EffectBase(EventDispatcher):
    """The base class for GLSL effects. It simply returns its input.

    See the module documentation for more details.

    """
    glsl = StringProperty(effect_trivial)
    'The glsl string defining your effect function. See the\n    module documentation for more details.\n\n    :attr:`glsl` is a :class:`~kivy.properties.StringProperty` and\n    defaults to\n    a trivial effect that returns its input.\n    '
    source = StringProperty('')
    "The (optional) filename from which to load the :attr:`glsl`\n    string.\n\n    :attr:`source` is a :class:`~kivy.properties.StringProperty` and\n    defaults to ''.\n    "
    fbo = ObjectProperty(None, allownone=True)
    'The fbo currently using this effect. The :class:`EffectBase`\n    automatically handles this.\n\n    :attr:`fbo` is an :class:`~kivy.properties.ObjectProperty` and\n    defaults to None.\n    '

    def __init__(self, *args, **kwargs):
        super(EffectBase, self).__init__(*args, **kwargs)
        fbind = self.fbind
        fbo_shader = self.set_fbo_shader
        fbind('fbo', fbo_shader)
        fbind('glsl', fbo_shader)
        fbind('source', self._load_from_source)

    def set_fbo_shader(self, *args):
        """Sets the :class:`~kivy.graphics.Fbo`'s shader by splicing
        the :attr:`glsl` string into a full fragment shader.

        The full shader is made up of :code:`shader_header +
        shader_uniforms + self.glsl + shader_footer_effect`.
        """
        if self.fbo is None:
            return
        self.fbo.set_fs(shader_header + shader_uniforms + self.glsl + shader_footer_effect)

    def _load_from_source(self, *args):
        """(internal) Loads the glsl string from a source file."""
        source = self.source
        if not source:
            return
        filename = resource_find(source)
        if filename is None:
            return Logger.error('Error reading file {filename}'.format(filename=source))
        with open(filename) as fileh:
            self.glsl = fileh.read()