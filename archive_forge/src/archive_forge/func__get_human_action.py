from typing import Optional
import time
from collections import defaultdict
import gym
from gym import spaces
import pyglet
import pyglet.window.key as key
def _get_human_action(self):
    """Read keyboard and mouse state for a new action"""
    action = {name: int(self.pressed_keys[key] if key is not None else None) for name, key in MINERL_ACTION_TO_KEYBOARD.items()}
    action['camera'] = self.last_mouse_delta
    self.last_mouse_delta = [0, 0]
    return action