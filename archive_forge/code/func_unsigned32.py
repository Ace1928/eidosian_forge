import platform
import unittest
import pygame
from pygame.locals import *
from pygame.pixelcopy import surface_to_array, map_array, array_to_surface, make_surface
def unsigned32(i):
    """cast signed 32 bit integer to an unsigned integer"""
    return i & 4294967295