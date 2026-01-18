import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_play_with_invalid_fade_ms(self):
    """Test playing a sound with invalid fade_ms."""
    with self.assertRaises(TypeError):
        self.sound.play(fade_ms='invalid')