import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
@unittest.skipIf(IS_PYPY, 'pypy skip')
def test_samples_address(self):
    """Test the _samples_address getter."""
    try:
        from ctypes import pythonapi, c_void_p, py_object
        Bytes_FromString = pythonapi.PyBytes_FromString
        Bytes_FromString.restype = c_void_p
        Bytes_FromString.argtypes = [py_object]
        samples = b'abcdefgh'
        sample_bytes = Bytes_FromString(samples)
        snd = mixer.Sound(buffer=samples)
        self.assertNotEqual(snd._samples_address, sample_bytes)
    finally:
        pygame.mixer.quit()
        with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
            snd._samples_address