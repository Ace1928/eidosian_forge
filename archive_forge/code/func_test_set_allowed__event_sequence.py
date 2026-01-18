import collections
import time
import unittest
import os
import pygame
def test_set_allowed__event_sequence(self):
    """Ensure a sequence of blocked event types can be unblocked/allowed."""
    event_types = [pygame.KEYDOWN, pygame.KEYUP, pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP]
    pygame.event.set_blocked(event_types)
    pygame.event.set_allowed(event_types)
    for etype in event_types:
        self.assertFalse(pygame.event.get_blocked(etype))