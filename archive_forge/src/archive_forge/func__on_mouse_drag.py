from typing import Optional
import time
from collections import defaultdict
import gym
from gym import spaces
import pyglet
import pyglet.window.key as key
def _on_mouse_drag(self, x, y, dx, dy, button, modifier):
    self.last_mouse_delta[0] -= dy * MOUSE_MULTIPLIER
    self.last_mouse_delta[1] += dx * MOUSE_MULTIPLIER