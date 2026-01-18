import collections
import time
import unittest
import os
import pygame
def test_post_blocked(self):
    """
        Test blocked events are not posted. Also test whether post()
        returns a boolean correctly
        """
    pygame.event.set_blocked(pygame.USEREVENT)
    self.assertFalse(pygame.event.post(pygame.event.Event(pygame.USEREVENT)))
    self.assertFalse(pygame.event.poll())
    pygame.event.set_allowed(pygame.USEREVENT)
    self.assertTrue(pygame.event.post(pygame.event.Event(pygame.USEREVENT)))
    self.assertEqual(pygame.event.poll(), pygame.event.Event(pygame.USEREVENT))