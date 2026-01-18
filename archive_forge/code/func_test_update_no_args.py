import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
def test_update_no_args(self):
    """does NOT update the display."""
    self.screen.fill('green')
    pygame.display.update()
    pygame.event.pump()
    self.question(f'Is the WHOLE screen green?')