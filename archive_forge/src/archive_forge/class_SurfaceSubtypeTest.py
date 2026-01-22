import os
import unittest
from pygame.tests import test_utils
from pygame.tests.test_utils import (
import pygame
from pygame.locals import *
from pygame.bufferproxy import BufferProxy
import platform
import gc
import weakref
import ctypes
class SurfaceSubtypeTest(unittest.TestCase):
    """Issue #280: Methods that return a new Surface preserve subclasses"""

    def setUp(self):
        pygame.display.init()

    def tearDown(self):
        pygame.display.quit()

    def test_copy(self):
        """Ensure method copy() preserves the surface's class

        When Surface is subclassed, the inherited copy() method will return
        instances of the subclass. Non Surface fields are uncopied, however.
        This includes instance attributes.
        """
        expected_size = (32, 32)
        ms1 = SurfaceSubclass(expected_size, SRCALPHA, 32)
        ms2 = ms1.copy()
        self.assertIsNot(ms1, ms2)
        self.assertIsInstance(ms1, pygame.Surface)
        self.assertIsInstance(ms2, pygame.Surface)
        self.assertIsInstance(ms1, SurfaceSubclass)
        self.assertIsInstance(ms2, SurfaceSubclass)
        self.assertTrue(ms1.test_attribute)
        self.assertRaises(AttributeError, getattr, ms2, 'test_attribute')
        self.assertEqual(ms2.get_size(), expected_size)

    def test_convert(self):
        """Ensure method convert() preserves the surface's class

        When Surface is subclassed, the inherited convert() method will return
        instances of the subclass. Non Surface fields are omitted, however.
        This includes instance attributes.
        """
        expected_size = (32, 32)
        ms1 = SurfaceSubclass(expected_size, 0, 24)
        ms2 = ms1.convert(24)
        self.assertIsNot(ms1, ms2)
        self.assertIsInstance(ms1, pygame.Surface)
        self.assertIsInstance(ms2, pygame.Surface)
        self.assertIsInstance(ms1, SurfaceSubclass)
        self.assertIsInstance(ms2, SurfaceSubclass)
        self.assertTrue(ms1.test_attribute)
        self.assertRaises(AttributeError, getattr, ms2, 'test_attribute')
        self.assertEqual(ms2.get_size(), expected_size)

    def test_convert_alpha(self):
        """Ensure method convert_alpha() preserves the surface's class

        When Surface is subclassed, the inherited convert_alpha() method will
        return instances of the subclass. Non Surface fields are omitted,
        however. This includes instance attributes.
        """
        pygame.display.set_mode((40, 40))
        expected_size = (32, 32)
        s = pygame.Surface(expected_size, SRCALPHA, 16)
        ms1 = SurfaceSubclass(expected_size, SRCALPHA, 32)
        ms2 = ms1.convert_alpha(s)
        self.assertIsNot(ms1, ms2)
        self.assertIsInstance(ms1, pygame.Surface)
        self.assertIsInstance(ms2, pygame.Surface)
        self.assertIsInstance(ms1, SurfaceSubclass)
        self.assertIsInstance(ms2, SurfaceSubclass)
        self.assertTrue(ms1.test_attribute)
        self.assertRaises(AttributeError, getattr, ms2, 'test_attribute')
        self.assertEqual(ms2.get_size(), expected_size)

    def test_subsurface(self):
        """Ensure method subsurface() preserves the surface's class

        When Surface is subclassed, the inherited subsurface() method will
        return instances of the subclass. Non Surface fields are uncopied,
        however. This includes instance attributes.
        """
        expected_size = (10, 12)
        ms1 = SurfaceSubclass((32, 32), SRCALPHA, 32)
        ms2 = ms1.subsurface((4, 5), expected_size)
        self.assertIsNot(ms1, ms2)
        self.assertIsInstance(ms1, pygame.Surface)
        self.assertIsInstance(ms2, pygame.Surface)
        self.assertIsInstance(ms1, SurfaceSubclass)
        self.assertIsInstance(ms2, SurfaceSubclass)
        self.assertTrue(ms1.test_attribute)
        self.assertRaises(AttributeError, getattr, ms2, 'test_attribute')
        self.assertEqual(ms2.get_size(), expected_size)