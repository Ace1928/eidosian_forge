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
class TestApp(App):

    def change_view(self, *l):
        self.sm.current = next(self.sm)

    def remove_screen(self, *l):
        self.sm.remove_widget(self.sm.get_screen('test1'))

    def build(self):
        root = FloatLayout()
        self.sm = sm = ScreenManager(transition=SwapTransition())
        sm.add_widget(Screen(name='test1'))
        sm.add_widget(Screen(name='test2'))
        btn = Button(size_hint=(None, None))
        btn.bind(on_release=self.change_view)
        btn2 = Button(size_hint=(None, None), x=100)
        btn2.bind(on_release=self.remove_screen)
        root.add_widget(sm)
        root.add_widget(btn)
        root.add_widget(btn2)
        return root