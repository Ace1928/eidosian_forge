import os
import platform
import unittest
import pygame
import time
def _type_error_checks(self, func_to_check):
    """Checks 3 TypeError (float, tuple, string) for the func_to_check"""
    'Intended for time.delay and time.wait functions'
    self.assertRaises(TypeError, func_to_check, 0.1)
    self.assertRaises(TypeError, pygame.time.delay, (0, 1))
    self.assertRaises(TypeError, pygame.time.delay, '10')