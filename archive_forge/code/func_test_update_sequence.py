import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
def test_update_sequence(self):
    """only updates the part of the display given by the rects."""
    self.screen.fill('green')
    rects = [pygame.Rect(0, 0, 100, 100), pygame.Rect(100, 0, 100, 100), pygame.Rect(200, 0, 100, 100), pygame.Rect(300, 300, 100, 100)]
    pygame.display.update(rects)
    pygame.event.pump()
    self.question(f'Is the screen green in {rects}?')