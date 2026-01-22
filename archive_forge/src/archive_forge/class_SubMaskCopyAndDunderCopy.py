from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
class SubMaskCopyAndDunderCopy(SubMaskDunderCopy):
    """Subclass of the Mask class to help test copying subclasses."""

    def copy(self):
        return super().copy()