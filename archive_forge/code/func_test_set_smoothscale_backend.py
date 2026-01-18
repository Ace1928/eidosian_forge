import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_set_smoothscale_backend(self):
    original_type = pygame.transform.get_smoothscale_backend()
    pygame.transform.set_smoothscale_backend('GENERIC')
    filter_type = pygame.transform.get_smoothscale_backend()
    self.assertEqual(filter_type, 'GENERIC')
    pygame.transform.set_smoothscale_backend(backend=original_type)

    def change():
        pygame.transform.set_smoothscale_backend('mmx')
    self.assertRaises(ValueError, change)

    def change():
        pygame.transform.set_smoothscale_backend(t='GENERIC')
    self.assertRaises(TypeError, change)

    def change():
        pygame.transform.set_smoothscale_backend(1)
    self.assertRaises(TypeError, change)
    if original_type != 'SSE':

        def change():
            pygame.transform.set_smoothscale_backend('SSE')
        self.assertRaises(ValueError, change)
    filter_type = pygame.transform.get_smoothscale_backend()
    self.assertEqual(filter_type, original_type)