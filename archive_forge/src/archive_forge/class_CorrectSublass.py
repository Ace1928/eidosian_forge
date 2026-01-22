import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
class CorrectSublass(mixer.Sound):

    def __init__(self, file):
        super().__init__(file=file)