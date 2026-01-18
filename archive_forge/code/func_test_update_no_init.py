import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
def test_update_no_init(self):
    """raises a pygame.error."""
    pygame.display.quit()
    with self.assertRaises(pygame.error):
        pygame.display.update()