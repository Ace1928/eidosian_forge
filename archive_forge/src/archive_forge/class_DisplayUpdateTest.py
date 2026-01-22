import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
class DisplayUpdateTest(unittest.TestCase):

    def question(self, qstr):
        """this is used in the interactive subclass."""

    def setUp(self):
        display.init()
        self.screen = pygame.display.set_mode((500, 500))
        self.screen.fill('black')
        pygame.display.flip()
        pygame.event.pump()

    def tearDown(self):
        display.quit()

    def test_update_negative(self):
        """takes rects with negative values."""
        self.screen.fill('green')
        r1 = pygame.Rect(0, 0, 100, 100)
        pygame.display.update(r1)
        r2 = pygame.Rect(-10, 0, 100, 100)
        pygame.display.update(r2)
        r3 = pygame.Rect(-10, 0, -100, -100)
        pygame.display.update(r3)
        self.question('Is the screen green in (0, 0, 100, 100)?')

    def test_update_sequence(self):
        """only updates the part of the display given by the rects."""
        self.screen.fill('green')
        rects = [pygame.Rect(0, 0, 100, 100), pygame.Rect(100, 0, 100, 100), pygame.Rect(200, 0, 100, 100), pygame.Rect(300, 300, 100, 100)]
        pygame.display.update(rects)
        pygame.event.pump()
        self.question(f'Is the screen green in {rects}?')

    def test_update_none_skipped(self):
        """None is skipped inside sequences."""
        self.screen.fill('green')
        rects = (None, pygame.Rect(100, 0, 100, 100), None, pygame.Rect(200, 0, 100, 100), pygame.Rect(300, 300, 100, 100))
        pygame.display.update(rects)
        pygame.event.pump()
        self.question(f'Is the screen green in {rects}?')

    def test_update_none(self):
        """does NOT update the display."""
        self.screen.fill('green')
        pygame.display.update(None)
        pygame.event.pump()
        self.question(f'Is the screen black and NOT green?')

    def test_update_no_args(self):
        """does NOT update the display."""
        self.screen.fill('green')
        pygame.display.update()
        pygame.event.pump()
        self.question(f'Is the WHOLE screen green?')

    def test_update_args(self):
        """updates the display using the args as a rect."""
        self.screen.fill('green')
        pygame.display.update(100, 100, 100, 100)
        pygame.event.pump()
        self.question('Is the screen green in (100, 100, 100, 100)?')

    def test_update_incorrect_args(self):
        """raises a ValueError when inputs are wrong."""
        with self.assertRaises(ValueError):
            pygame.display.update(100, 'asdf', 100, 100)
        with self.assertRaises(ValueError):
            pygame.display.update([100, 'asdf', 100, 100])

    def test_update_no_init(self):
        """raises a pygame.error."""
        pygame.display.quit()
        with self.assertRaises(pygame.error):
            pygame.display.update()