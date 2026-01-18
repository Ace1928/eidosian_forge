import collections
import time
import unittest
import os
import pygame
def test_post_large_user_event(self):
    pygame.event.post(pygame.event.Event(pygame.USEREVENT, {'a': 'a' * 1024}, test=list(range(100))))
    e = pygame.event.poll()
    self.assertEqual(e.type, pygame.USEREVENT)
    self.assertEqual(e.a, 'a' * 1024)
    self.assertEqual(e.test, list(range(100)))