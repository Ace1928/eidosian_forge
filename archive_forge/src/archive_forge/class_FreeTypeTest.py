import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
class FreeTypeTest(unittest.TestCase):

    def setUp(self):
        ft.init()

    def tearDown(self):
        ft.quit()

    def test_resolution(self):
        try:
            ft.set_default_resolution()
            resolution = ft.get_default_resolution()
            self.assertEqual(resolution, 72)
            new_resolution = resolution + 10
            ft.set_default_resolution(new_resolution)
            self.assertEqual(ft.get_default_resolution(), new_resolution)
            ft.init(resolution=resolution + 20)
            self.assertEqual(ft.get_default_resolution(), new_resolution)
        finally:
            ft.set_default_resolution()

    def test_autoinit_and_autoquit(self):
        pygame.init()
        self.assertTrue(ft.get_init())
        pygame.quit()
        self.assertFalse(ft.get_init())
        pygame.init()
        self.assertTrue(ft.get_init())
        pygame.quit()
        self.assertFalse(ft.get_init())

    def test_init(self):
        ft.quit()
        ft.init()
        self.assertTrue(ft.get_init())

    def test_init__multiple(self):
        ft.init()
        ft.init()
        self.assertTrue(ft.get_init())

    def test_quit(self):
        ft.quit()
        self.assertFalse(ft.get_init())

    def test_quit__multiple(self):
        ft.quit()
        ft.quit()
        self.assertFalse(ft.get_init())

    def test_get_init(self):
        self.assertTrue(ft.get_init())

    def test_cache_size(self):
        DEFAULT_CACHE_SIZE = 64
        self.assertEqual(ft.get_cache_size(), DEFAULT_CACHE_SIZE)
        ft.quit()
        self.assertEqual(ft.get_cache_size(), 0)
        new_cache_size = DEFAULT_CACHE_SIZE * 2
        ft.init(cache_size=new_cache_size)
        self.assertEqual(ft.get_cache_size(), new_cache_size)

    def test_get_error(self):
        """Ensures get_error() is initially empty (None)."""
        error_msg = ft.get_error()
        self.assertIsNone(error_msg)

    def test_get_version(self):
        ft.quit()
        self.assertIsNotNone(ft.get_version(linked=False))
        self.assertIsNotNone(ft.get_version(linked=True))