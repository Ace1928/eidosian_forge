import collections
import time
import unittest
import os
import pygame
def test_peek__event_sequence(self):
    """Ensure peek() can handle a sequence of event types."""
    event_types = [pygame.KEYDOWN, pygame.KEYUP, pygame.MOUSEMOTION]
    other_event_type = pygame.MOUSEBUTTONUP
    pygame.event.clear()
    peeked = pygame.event.peek(event_types)
    self.assertFalse(peeked)
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(other_event_type, **EVENT_TEST_PARAMS[other_event_type]))
    peeked = pygame.event.peek(event_types)
    self.assertFalse(peeked)
    pygame.event.clear()
    pygame.event.post(pygame.event.Event(event_types[0], **EVENT_TEST_PARAMS[event_types[0]]))
    peeked = pygame.event.peek(event_types)
    self.assertTrue(peeked)
    pygame.event.clear()
    for etype in event_types:
        pygame.event.post(pygame.event.Event(etype, **EVENT_TEST_PARAMS[etype]))
    peeked = pygame.event.peek(event_types)
    self.assertTrue(peeked)