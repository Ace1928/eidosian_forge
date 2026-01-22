import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
class _ObjectWithRectAttribute:

    def __init__(self, r):
        self.rect = r