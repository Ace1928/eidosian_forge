from math import cos, sin, pi, sqrt, atan
from colorsys import rgb_to_hsv, hsv_to_rgb
from kivy.clock import Clock
from kivy.graphics import Mesh, InstructionGroup, Color
from kivy.logger import Logger
from kivy.properties import (NumericProperty, BoundedNumericProperty,
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.widget import Widget
from kivy.utils import get_color_from_hex, get_hex_from_color
class _ColorArc(InstructionGroup):

    def __init__(self, r_min, r_max, theta_min, theta_max, color=(0, 0, 1, 1), origin=(0, 0), **kwargs):
        super(_ColorArc, self).__init__(**kwargs)
        self.origin = origin
        self.r_min = r_min
        self.r_max = r_max
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.color = color
        self.color_instr = Color(*color, mode='hsv')
        self.add(self.color_instr)
        self.mesh = self.get_mesh()
        self.add(self.mesh)

    def __str__(self):
        return 'r_min: %s r_max: %s theta_min: %s theta_max: %s color: %s' % (self.r_min, self.r_max, self.theta_min, self.theta_max, self.color)

    def get_mesh(self):
        v = []
        theta_step_outer = 0.1
        theta = self.theta_max - self.theta_min
        d_outer = int(theta / theta_step_outer)
        theta_step_outer = theta / d_outer
        if self.r_min == 0:
            for x in range(0, d_outer, 2):
                v += polar_to_rect(self.origin, self.r_max, self.theta_min + x * theta_step_outer) * 2
                v += polar_to_rect(self.origin, 0, 0) * 2
                v += polar_to_rect(self.origin, self.r_max, self.theta_min + (x + 1) * theta_step_outer) * 2
            if not d_outer & 1:
                v += polar_to_rect(self.origin, self.r_max, self.theta_min + d_outer * theta_step_outer) * 2
        else:
            for x in range(d_outer + 1):
                v += polar_to_rect(self.origin, self.r_min, self.theta_min + x * theta_step_outer) * 2
                v += polar_to_rect(self.origin, self.r_max, self.theta_min + x * theta_step_outer) * 2
        return Mesh(vertices=v, indices=range(int(len(v) / 4)), mode='triangle_strip')

    def change_color(self, color=None, color_delta=None, sv=None, a=None):
        self.remove(self.color_instr)
        if color is not None:
            self.color = color
        elif color_delta is not None:
            self.color = [self.color[i] + color_delta[i] for i in range(4)]
        elif sv is not None:
            self.color = (self.color[0], sv[0], sv[1], self.color[3])
        elif a is not None:
            self.color = (self.color[0], self.color[1], self.color[2], a)
        self.color_instr = Color(*self.color, mode='hsv')
        self.insert(0, self.color_instr)