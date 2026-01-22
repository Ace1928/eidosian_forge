from typing import Optional
import time
from collections import defaultdict
import gym
from gym import spaces
import pyglet
import pyglet.window.key as key
def _update_image(self, arr):
    self.window.switch_to()
    image = pyglet.image.ImageData(arr.shape[1], arr.shape[0], 'RGB', arr.tobytes(), pitch=arr.shape[1] * -3)
    texture = image.get_texture()
    texture.blit(0, 0)
    self.window.flip()