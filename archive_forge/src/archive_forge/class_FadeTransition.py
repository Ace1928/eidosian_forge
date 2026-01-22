from kivy.compat import iteritems
from kivy.logger import Logger
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import (StringProperty, ObjectProperty, AliasProperty,
from kivy.animation import Animation, AnimationTransition
from kivy.uix.relativelayout import RelativeLayout
from kivy.lang import Builder
from kivy.graphics import (RenderContext, Rectangle, Fbo,
class FadeTransition(ShaderTransition):
    """Fade transition, based on a fragment Shader.
    """
    FADE_TRANSITION_FS = '$HEADER$\n    uniform float t;\n    uniform sampler2D tex_in;\n    uniform sampler2D tex_out;\n\n    void main(void) {\n        vec4 cin = vec4(texture2D(tex_in, tex_coord0.st));\n        vec4 cout = vec4(texture2D(tex_out, tex_coord0.st));\n        vec4 frag_col = vec4(t * cin) + vec4((1.0 - t) * cout);\n        gl_FragColor = frag_col;\n    }\n    '
    fs = StringProperty(FADE_TRANSITION_FS)