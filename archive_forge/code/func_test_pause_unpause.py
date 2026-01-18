import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_pause_unpause(self):
    """
        Test if the Channel can be paused and unpaused.
        """
    if mixer.get_init() is None:
        mixer.init()
    sound = pygame.mixer.Sound(example_path('data/house_lo.wav'))
    channel = sound.play()
    channel.pause()
    self.assertTrue(channel.get_busy(), msg="Channel should be paused but it's not.")
    channel.unpause()
    self.assertTrue(channel.get_busy(), msg="Channel should be unpaused but it's not.")
    sound.stop()