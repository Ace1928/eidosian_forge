import os
import sys
import platform
import unittest
import time
from pygame.tests.test_utils import example_path
import pygame
issue #955. unload music whenever mixer.quit() is called