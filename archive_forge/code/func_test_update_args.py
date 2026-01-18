import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
def test_update_args(self):
    """updates the display using the args as a rect."""
    self.screen.fill('green')
    pygame.display.update(100, 100, 100, 100)
    pygame.event.pump()
    self.question('Is the screen green in (100, 100, 100, 100)?')