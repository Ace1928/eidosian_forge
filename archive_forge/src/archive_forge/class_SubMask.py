from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
class SubMask(pygame.mask.Mask):
    """Subclass of the Mask class to help test subclassing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_attribute = True