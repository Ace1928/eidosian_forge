import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
class ChannelSetVolumeTest(unittest.TestCase):

    def setUp(self):
        mixer.init()
        self.channel = pygame.mixer.Channel(0)
        self.sound = pygame.mixer.Sound(example_path('data/boom.wav'))

    def tearDown(self):
        mixer.quit()

    def test_set_volume_with_one_argument(self):
        self.channel.play(self.sound)
        self.channel.set_volume(0.5)
        self.assertEqual(self.channel.get_volume(), 0.5)

    @unittest.expectedFailure
    def test_set_volume_with_two_arguments(self):
        self.channel.play(self.sound)
        self.channel.set_volume(0.3, 0.7)
        self.assertEqual(self.channel.get_volume(), (0.3, 0.7))