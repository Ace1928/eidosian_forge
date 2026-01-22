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
class ShaderTransition(TransitionBase):
    '''Transition class that uses a Shader for animating the transition between
    2 screens. By default, this class doesn't assign any fragment/vertex
    shader. If you want to create your own fragment shader for the transition,
    you need to declare the header yourself and include the "t", "tex_in" and
    "tex_out" uniform::

        # Create your own transition. This shader implements a "fading"
        # transition.
        fs = """$HEADER
            uniform float t;
            uniform sampler2D tex_in;
            uniform sampler2D tex_out;

            void main(void) {
                vec4 cin = texture2D(tex_in, tex_coord0);
                vec4 cout = texture2D(tex_out, tex_coord0);
                gl_FragColor = mix(cout, cin, t);
            }
        """

        # And create your transition
        tr = ShaderTransition(fs=fs)
        sm = ScreenManager(transition=tr)

    '''
    fs = StringProperty(None)
    'Fragment shader to use.\n\n    :attr:`fs` is a :class:`~kivy.properties.StringProperty` and defaults to\n    None.'
    vs = StringProperty(None)
    'Vertex shader to use.\n\n    :attr:`vs` is a :class:`~kivy.properties.StringProperty` and defaults to\n    None.'
    clearcolor = ColorProperty([0, 0, 0, 1])
    'Sets the color of Fbo ClearColor.\n\n    .. versionadded:: 1.9.0\n\n    :attr:`clearcolor` is a :class:`~kivy.properties.ColorProperty`\n    and defaults to [0, 0, 0, 1].\n\n    .. versionchanged:: 2.0.0\n        Changed from :class:`~kivy.properties.ListProperty` to\n        :class:`~kivy.properties.ColorProperty`.\n    '

    def make_screen_fbo(self, screen):
        fbo = Fbo(size=screen.size, with_stencilbuffer=True)
        with fbo:
            ClearColor(*self.clearcolor)
            ClearBuffers()
        fbo.add(screen.canvas)
        with fbo.before:
            PushMatrix()
            Translate(-screen.x, -screen.y, 0)
        with fbo.after:
            PopMatrix()
        return fbo

    def on_progress(self, progress):
        self.render_ctx['t'] = progress

    def on_complete(self):
        self.render_ctx['t'] = 1.0
        super(ShaderTransition, self).on_complete()

    def _remove_out_canvas(self, *args):
        if self.screen_out and self.screen_out.canvas in self.manager.canvas.children and (self.screen_out not in self.manager.children):
            self.manager.canvas.remove(self.screen_out.canvas)

    def add_screen(self, screen):
        self.screen_in.pos = self.screen_out.pos
        self.screen_in.size = self.screen_out.size
        self.manager.real_remove_widget(self.screen_out)
        self.manager.canvas.add(self.screen_out.canvas)

        def remove_screen_out(instr):
            Clock.schedule_once(self._remove_out_canvas, -1)
            self.render_ctx.remove(instr)
        self.fbo_in = self.make_screen_fbo(self.screen_in)
        self.fbo_out = self.make_screen_fbo(self.screen_out)
        self.manager.canvas.add(self.fbo_in)
        self.manager.canvas.add(self.fbo_out)
        self.render_ctx = RenderContext(fs=self.fs, vs=self.vs, use_parent_modelview=True, use_parent_projection=True)
        with self.render_ctx:
            BindTexture(texture=self.fbo_out.texture, index=1)
            BindTexture(texture=self.fbo_in.texture, index=2)
            x, y = self.screen_in.pos
            w, h = self.fbo_in.texture.size
            Rectangle(size=(w, h), pos=(x, y), tex_coords=self.fbo_in.texture.tex_coords)
            Callback(remove_screen_out)
        self.render_ctx['tex_out'] = 1
        self.render_ctx['tex_in'] = 2
        self.manager.canvas.add(self.render_ctx)

    def remove_screen(self, screen):
        self.manager.canvas.remove(self.fbo_in)
        self.manager.canvas.remove(self.fbo_out)
        self.manager.canvas.remove(self.render_ctx)
        self._remove_out_canvas()
        self.manager.real_add_widget(self.screen_in)

    def stop(self):
        self._remove_out_canvas()
        super(ShaderTransition, self).stop()