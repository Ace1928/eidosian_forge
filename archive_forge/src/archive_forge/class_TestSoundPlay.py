import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
class TestSoundPlay(unittest.TestCase):

    def setUp(self):
        mixer.init()
        self.filename = example_path(os.path.join('data', 'house_lo.wav'))
        self.sound = mixer.Sound(file=self.filename)

    def tearDown(self):
        mixer.quit()

    def test_play_once(self):
        """Test playing a sound once."""
        channel = self.sound.play()
        self.assertIsInstance(channel, pygame.mixer.Channel)
        self.assertTrue(channel.get_busy())

    def test_play_multiple_times(self):
        """Test playing a sound multiple times."""
        frequency, format, channels = mixer.get_init()
        sound_length_in_ms = 100
        bytes_per_ms = int(frequency / 1000 * channels * (abs(format) // 8))
        sound = mixer.Sound(b'\x00' * int(sound_length_in_ms * bytes_per_ms))
        self.assertAlmostEqual(sound.get_length(), sound_length_in_ms / 1000.0, places=2)
        num_loops = 5
        channel = sound.play(loops=num_loops)
        self.assertIsInstance(channel, pygame.mixer.Channel)
        pygame.time.wait(sound_length_in_ms * num_loops - 100)
        self.assertTrue(channel.get_busy())
        pygame.time.wait(sound_length_in_ms + 200)
        self.assertFalse(channel.get_busy())

    def test_play_indefinitely(self):
        """Test playing a sound indefinitely."""
        frequency, format, channels = mixer.get_init()
        sound_length_in_ms = 100
        bytes_per_ms = int(frequency / 1000 * channels * (abs(format) // 8))
        sound = mixer.Sound(b'\x00' * int(sound_length_in_ms * bytes_per_ms))
        channel = sound.play(loops=-1)
        self.assertIsInstance(channel, pygame.mixer.Channel)
        for _ in range(2):
            self.assertTrue(channel.get_busy())
            pygame.time.wait(sound_length_in_ms)

    def test_play_with_maxtime(self):
        """Test playing a sound with maxtime."""
        channel = self.sound.play(maxtime=200)
        self.assertIsInstance(channel, pygame.mixer.Channel)
        self.assertTrue(channel.get_busy())
        pygame.time.wait(200 + 50)
        self.assertFalse(channel.get_busy())

    def test_play_with_fade_ms(self):
        """Test playing a sound with fade_ms."""
        channel = self.sound.play(fade_ms=500)
        self.assertIsInstance(channel, pygame.mixer.Channel)
        self.assertTrue(channel.get_busy())
        pygame.time.wait(250)
        self.assertGreater(channel.get_volume(), 0.3)
        self.assertLess(channel.get_volume(), 0.8)
        pygame.time.wait(300)
        self.assertEqual(channel.get_volume(), 1.0)

    def test_play_with_invalid_loops(self):
        """Test playing a sound with invalid loops."""
        with self.assertRaises(TypeError):
            self.sound.play(loops='invalid')

    def test_play_with_invalid_maxtime(self):
        """Test playing a sound with invalid maxtime."""
        with self.assertRaises(TypeError):
            self.sound.play(maxtime='invalid')

    def test_play_with_invalid_fade_ms(self):
        """Test playing a sound with invalid fade_ms."""
        with self.assertRaises(TypeError):
            self.sound.play(fade_ms='invalid')