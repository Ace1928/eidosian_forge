import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_sound_unicode(self):
    """test non-ASCII unicode path"""
    mixer.init()
    import shutil
    ep = example_path('data')
    temp_file = os.path.join(ep, '你好.wav')
    org_file = os.path.join(ep, 'house_lo.wav')
    shutil.copy(org_file, temp_file)
    try:
        with open(temp_file, 'rb') as f:
            pass
    except OSError:
        raise unittest.SkipTest('the path cannot be opened')
    try:
        sound = mixer.Sound(temp_file)
        del sound
    finally:
        os.remove(temp_file)