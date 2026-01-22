from the Sprite and AbstractGroup classes below, it is strongly recommended
from weakref import WeakSet
from warnings import warn
import pygame
from pygame.rect import Rect
from pygame.time import get_ticks
from pygame.mask import from_surface
class WeakDirtySprite(WeakSprite, DirtySprite):
    """A subclass of WeakSprite and DirtySprite that combines the benefits
    of both classes.
    """