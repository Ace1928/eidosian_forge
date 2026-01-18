import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_threshold_set_behavior0(self):
    """raises an error when set_behavior=1
        and set_color is not None,
        and dest_surf is not None.
        """
    from pygame.transform import threshold
    s1 = pygame.Surface((32, 32), SRCALPHA, 32)
    s2 = pygame.Surface((32, 32), SRCALPHA, 32)
    THRESHOLD_BEHAVIOR_COUNT = 0
    self.assertRaises(TypeError, threshold, dest_surface=None, surface=s2, search_color=(30, 30, 30), threshold=(11, 11, 11), set_color=(0, 0, 0), set_behavior=THRESHOLD_BEHAVIOR_COUNT)
    self.assertRaises(TypeError, threshold, dest_surface=s1, surface=s2, search_color=(30, 30, 30), threshold=(11, 11, 11), set_color=None, set_behavior=THRESHOLD_BEHAVIOR_COUNT)
    threshold(dest_surface=None, surface=s2, search_color=(30, 30, 30), threshold=(11, 11, 11), set_color=None, set_behavior=THRESHOLD_BEHAVIOR_COUNT)