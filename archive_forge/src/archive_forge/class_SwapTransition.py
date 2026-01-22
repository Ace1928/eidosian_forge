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
class SwapTransition(TransitionBase):
    """Swap transition that looks like iOS transition when a new window
    appears on the screen.
    """

    def __init__(self, **kwargs):
        super(SwapTransition, self).__init__(**kwargs)
        self.scales = {}

    def start(self, manager):
        for screen in (self.screen_in, self.screen_out):
            with screen.canvas.before:
                PushMatrix(group='swaptransition_scale')
                scale = Scale(group='swaptransition_scale')
            with screen.canvas.after:
                PopMatrix(group='swaptransition_scale')
            screen.bind(center=self.update_scale)
            self.scales[screen] = scale
        super(SwapTransition, self).start(manager)

    def update_scale(self, screen, center):
        self.scales[screen].origin = center

    def add_screen(self, screen):
        self.manager.real_add_widget(screen, 1)

    def on_complete(self):
        self.screen_in.pos = self.manager.pos
        self.screen_out.pos = self.manager.pos
        for screen in (self.screen_in, self.screen_out):
            for canvas in (screen.canvas.before, screen.canvas.after):
                canvas.remove_group('swaptransition_scale')
        super(SwapTransition, self).on_complete()

    def on_progress(self, progression):
        a = self.screen_in
        b = self.screen_out
        manager = self.manager
        self.scales[b].xyz = [1.0 - progression * 0.7 for xyz in 'xyz']
        self.scales[a].xyz = [0.5 + progression * 0.5 for xyz in 'xyz']
        a.center_y = b.center_y = manager.center_y
        al = AnimationTransition.in_out_sine
        if progression < 0.5:
            p2 = al(progression * 2)
            width = manager.width * 0.7
            widthb = manager.width * 0.2
            a.x = manager.center_x + p2 * width / 2.0
            b.center_x = manager.center_x - p2 * widthb / 2.0
        else:
            if self.screen_in is self.manager.children[-1]:
                self.manager.real_remove_widget(self.screen_in)
                self.manager.real_add_widget(self.screen_in)
            p2 = al((progression - 0.5) * 2)
            width = manager.width * 0.85
            widthb = manager.width * 0.2
            a.x = manager.x + width * (1 - p2)
            b.center_x = manager.center_x - (1 - p2) * widthb / 2.0