import os
import sys
import platform
import unittest
import time
from pygame.tests.test_utils import example_path
import pygame
def test_unload(self):
    import shutil
    import tempfile
    ep = example_path('data')
    org_file = os.path.join(ep, 'house_lo.wav')
    tmpfd, tmppath = tempfile.mkstemp('.wav')
    os.close(tmpfd)
    shutil.copy(org_file, tmppath)
    try:
        pygame.mixer.music.load(tmppath)
        pygame.mixer.music.unload()
    finally:
        os.remove(tmppath)