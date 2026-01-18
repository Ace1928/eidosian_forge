import collections
import time
import unittest
import os
import pygame
def test_set_blocked(self):
    """Ensure events can be blocked from the queue."""
    event = EVENT_TYPES[0]
    unblocked_event = EVENT_TYPES[1]
    pygame.event.set_blocked(event)
    self.assertTrue(pygame.event.get_blocked(event))
    self.assertFalse(pygame.event.get_blocked(unblocked_event))
    posted = pygame.event.post(pygame.event.Event(event, **EVENT_TEST_PARAMS[event]))
    self.assertFalse(posted)
    posted = pygame.event.post(pygame.event.Event(unblocked_event, **EVENT_TEST_PARAMS[unblocked_event]))
    self.assertTrue(posted)
    ret = pygame.event.get()
    should_be_blocked = [e for e in ret if e.type == event]
    should_be_allowed_types = [e.type for e in ret if e.type != event]
    self.assertEqual(should_be_blocked, [])
    self.assertTrue(unblocked_event in should_be_allowed_types)