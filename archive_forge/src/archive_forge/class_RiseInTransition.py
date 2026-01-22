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
class RiseInTransition(ShaderTransition):
    """Transition where the new screen rises from the screen centre,
    becoming larger and changing from transparent to opaque until it
    fills the screen. Mimics the popular/standard Android transition.

    .. versionadded:: 1.8.0
    """
    duration = NumericProperty(0.2)
    'Duration in seconds of the transition, replacing the default of\n    :class:`TransitionBase`.\n\n    :class:`duration` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to .2 (= 200ms).\n    '
    RISEIN_TRANSITION_FS = "$HEADER$\n    uniform float t;\n    uniform sampler2D tex_in;\n    uniform sampler2D tex_out;\n\n    void main(void) {\n        /* quantities for position and opacity calculation */\n        float tr = 0.5 - 0.5*sqrt(sin(t));  /* 'real' time */\n        vec2 diff = (tex_coord0.st - 0.5) * (1.0/(1.0-tr));\n        vec2 dist = diff + 0.5;\n        float max_dist = 1.0 - tr;\n\n        /* in and out colors */\n        vec4 cin = vec4(texture2D(tex_in, dist));\n        vec4 cout = vec4(texture2D(tex_out, tex_coord0.st));\n\n        /* opacities for in and out textures */\n        float oin = clamp(sin(2.0*t), 0.0, 1.0);\n        float oout = clamp(1.0 - sin(2.0*t), 0.0, 1.0);\n\n        bvec2 outside_bounds = bvec2(abs(tex_coord0.s - 0.5) > 0.5*max_dist,\n                                     abs(tex_coord0.t - 0.5) > 0.5*max_dist);\n\n        vec4 frag_col;\n        if (any(outside_bounds) ){\n            frag_col = vec4(cout.x, cout.y, cout.z, 1.0);\n            }\n        else {\n            frag_col = vec4(oout*cout.x + oin*cin.x, oout*cout.y + oin*cin.y,\n                            oout*cout.z + oin*cin.z, 1.0);\n            }\n\n        gl_FragColor = frag_col;\n    }\n    "
    fs = StringProperty(RISEIN_TRANSITION_FS)