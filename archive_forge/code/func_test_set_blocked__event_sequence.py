import collections
import time
import unittest
import os
import pygame
def test_set_blocked__event_sequence(self):
    """Ensure a sequence of event types can be blocked."""
    event_types = [pygame.KEYDOWN, pygame.KEYUP, pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.WINDOWFOCUSLOST, pygame.USEREVENT]
    pygame.event.set_blocked(event_types)
    for etype in event_types:
        self.assertTrue(pygame.event.get_blocked(etype))