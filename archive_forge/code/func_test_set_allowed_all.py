import collections
import time
import unittest
import os
import pygame
def test_set_allowed_all(self):
    """Ensure all events can be unblocked/allowed at once."""
    pygame.event.set_blocked(None)
    for e in EVENT_TYPES:
        self.assertTrue(pygame.event.get_blocked(e))
    pygame.event.set_allowed(None)
    for e in EVENT_TYPES:
        self.assertFalse(pygame.event.get_blocked(e))