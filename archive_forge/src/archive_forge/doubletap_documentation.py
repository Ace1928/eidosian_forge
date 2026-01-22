from time import time
from kivy.config import Config
from kivy.vector import Vector
Find a double tap touch within self.touches.
        The touch must be not a previous double tap and the distance must be
        within the specified threshold. Additionally, the touch profiles
        must be the same kind of touch.
        