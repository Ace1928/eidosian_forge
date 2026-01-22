import sys
import os
from pygame.base import *  # pylint: disable=wildcard-import; lgtm[py/polluting-import]
from pygame.constants import *  # now has __all__ pylint: disable=wildcard-import; lgtm[py/polluting-import]
from pygame.version import *  # pylint: disable=wildcard-import; lgtm[py/polluting-import]
from pygame.rect import Rect
from pygame.rwobject import encode_string, encode_file_path
import pygame.surflock
import pygame.color
import pygame.bufferproxy
import pygame.math
import copyreg
def __color_constructor(r, g, b, a):
    return Color(r, g, b, a)