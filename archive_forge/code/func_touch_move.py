from kivy.uix.scrollview import ScrollView
from kivy.properties import ObjectProperty, NumericProperty, BooleanProperty
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.config import Config
def touch_move(instance, touch):
    instance.center = touch.pos