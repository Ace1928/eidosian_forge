from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
def add_internal(self, sprite, layer=None):
    if self.__sprite is not None:
        self.__sprite.remove_internal(self)
        self.remove_internal(self.__sprite)
    self.__sprite = sprite