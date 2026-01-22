from kivy.clock import Clock
from kivy.uix.relativelayout import RelativeLayout
from kivy.properties import (StringProperty, ObjectProperty, ListProperty,
from kivy.graphics import (RenderContext, Fbo, Color, Rectangle,
from kivy.event import EventDispatcher
from kivy.base import EventLoop
from kivy.resources import resource_find
from kivy.logger import Logger
def _update_glsl(self, *largs):
    """(internal) Passes new time and resolution uniform
        variables to the shader.
        """
    time = Clock.get_boottime()
    resolution = [float(size) for size in self.size]
    self.canvas['time'] = time
    self.canvas['resolution'] = resolution
    for fbo in self.fbo_list:
        fbo['time'] = time
        fbo['resolution'] = resolution