import collections
import time
import unittest
import os
import pygame
def test_post_and_get_keydown(self):
    """Ensure keydown events can be posted to the queue."""
    activemodkeys = pygame.key.get_mods()
    events = [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p), pygame.event.Event(pygame.KEYDOWN, key=pygame.K_y, mod=activemodkeys), pygame.event.Event(pygame.KEYDOWN, key=pygame.K_g, unicode='g'), pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a, unicode=None), pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m, mod=None, window=None), pygame.event.Event(pygame.KEYDOWN, key=pygame.K_e, mod=activemodkeys, unicode='e')]
    for e in events:
        pygame.event.post(e)
        posted_event = pygame.event.poll()
        self.assertEqual(e, posted_event, race_condition_notification)