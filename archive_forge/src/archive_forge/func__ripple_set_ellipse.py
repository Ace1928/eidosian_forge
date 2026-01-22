from kivy.animation import Animation
from kivy.clock import Clock
from kivy.graphics import CanvasBase, Color, Ellipse, ScissorPush, ScissorPop
from kivy.properties import BooleanProperty, ListProperty, NumericProperty, \
from kivy.uix.relativelayout import RelativeLayout
def _ripple_set_ellipse(self, instance, value):
    ellipse = self.ripple_ellipse
    if not ellipse:
        return
    ripple_pos = self.ripple_pos
    ripple_rad = self.ripple_rad
    ellipse.size = (ripple_rad, ripple_rad)
    ellipse.pos = (ripple_pos[0] - ripple_rad / 2.0, ripple_pos[1] - ripple_rad / 2.0)