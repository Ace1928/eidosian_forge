import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_undefined_character_code(self):
    font = self._TEST_FONTS['sans']
    img, size1 = font.render(chr(1), (0, 0, 0), size=24)
    img, size0 = font.render('', (0, 0, 0), size=24)
    self.assertTrue(size1.width > size0.width)
    metrics = font.get_metrics(chr(1) + chr(48), size=24)
    self.assertEqual(len(metrics), 2)
    self.assertIsNone(metrics[0])
    self.assertIsInstance(metrics[1], tuple)