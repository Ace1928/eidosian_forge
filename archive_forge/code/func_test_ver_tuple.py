from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_ver_tuple(ver):
    self.assertIsInstance(ver, tuple)
    self.assertEqual(len(ver), 3)
    for i in ver:
        self.assertIsInstance(i, int)