import weakref
from functools import partial
from itertools import chain
from kivy.animation import Animation
from kivy.logger import Logger
from kivy.graphics.transformation import Matrix
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.weakproxy import WeakProxy
from kivy.properties import (
def toggle_position(self, button):
    to_bottom = button.text == 'Move to Bottom'
    if to_bottom:
        button.text = 'Move to Top'
        if self.widget_info:
            Animation(top=250, t='out_quad', d=0.3).start(self.layout)
        else:
            Animation(top=60, t='out_quad', d=0.3).start(self.layout)
        bottom_bar = self.layout.children[1]
        self.layout.remove_widget(bottom_bar)
        self.layout.add_widget(bottom_bar)
    else:
        button.text = 'Move to Bottom'
        if self.widget_info:
            Animation(top=self.height, t='out_quad', d=0.3).start(self.layout)
        else:
            Animation(y=self.height - 60, t='out_quad', d=0.3).start(self.layout)
        bottom_bar = self.layout.children[1]
        self.layout.remove_widget(bottom_bar)
        self.layout.add_widget(bottom_bar)
    self.at_bottom = to_bottom