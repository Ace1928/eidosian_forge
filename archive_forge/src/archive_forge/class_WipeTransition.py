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
class WipeTransition(ShaderTransition):
    """Wipe transition, based on a fragment Shader.
    """
    WIPE_TRANSITION_FS = '$HEADER$\n    uniform float t;\n    uniform sampler2D tex_in;\n    uniform sampler2D tex_out;\n\n    void main(void) {\n        vec4 cin = texture2D(tex_in, tex_coord0);\n        vec4 cout = texture2D(tex_out, tex_coord0);\n        gl_FragColor = mix(cout, cin, clamp((-1.5 + 1.5*tex_coord0.x + 2.5*t),\n            0.0, 1.0));\n    }\n    '
    fs = StringProperty(WIPE_TRANSITION_FS)