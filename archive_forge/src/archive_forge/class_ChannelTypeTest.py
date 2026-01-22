import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
class ChannelTypeTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mixer.init()

    @classmethod
    def tearDownClass(cls):
        mixer.quit()

    def setUp(cls):
        if mixer.get_init() is None:
            mixer.init()

    def test_channel(self):
        """Ensure Channel() creation works."""
        channel = mixer.Channel(0)
        self.assertIsInstance(channel, mixer.ChannelType)
        self.assertEqual(channel.__class__.__name__, 'Channel')

    def test_channel__without_arg(self):
        """Ensure exception for Channel() creation with no argument."""
        with self.assertRaises(TypeError):
            mixer.Channel()

    def test_channel__invalid_id(self):
        """Ensure exception for Channel() creation with an invalid id."""
        with self.assertRaises(IndexError):
            mixer.Channel(-1)

    def test_channel__before_init(self):
        """Ensure exception for Channel() creation with non-init mixer."""
        mixer.quit()
        with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
            mixer.Channel(0)

    def test_fadeout(self):
        """Ensure Channel.fadeout() stops playback after fading out."""
        channel = mixer.Channel(0)
        sound = mixer.Sound(example_path('data/house_lo.wav'))
        channel.play(sound)
        fadeout_time = 1000
        channel.fadeout(fadeout_time)
        pygame.time.wait(fadeout_time + 100)
        self.assertFalse(channel.get_busy())

    def test_get_busy(self):
        """Ensure an idle channel's busy state is correct."""
        expected_busy = False
        channel = mixer.Channel(0)
        busy = channel.get_busy()
        self.assertEqual(busy, expected_busy)

    def test_get_busy__active(self):
        """Ensure an active channel's busy state is correct."""
        channel = mixer.Channel(0)
        sound = mixer.Sound(example_path('data/house_lo.wav'))
        channel.play(sound)
        self.assertTrue(channel.get_busy())

    def todo_test_get_endevent(self):
        self.fail()

    def test_get_queue(self):
        """Ensure Channel.get_queue() returns any queued Sound."""
        channel = mixer.Channel(0)
        frequency, format, channels = mixer.get_init()
        sound_length_in_ms = 200
        sound_length_in_ms_2 = 400
        bytes_per_ms = int(frequency / 1000 * channels * (abs(format) // 8))
        sound1 = mixer.Sound(b'\x00' * int(sound_length_in_ms * bytes_per_ms))
        sound2 = mixer.Sound(b'\x00' * int(sound_length_in_ms_2 * bytes_per_ms))
        channel.play(sound1)
        channel.queue(sound2)
        self.assertEqual(channel.get_queue().get_length(), sound2.get_length())
        pygame.time.wait(sound_length_in_ms + 100)
        self.assertEqual(channel.get_sound().get_length(), sound2.get_length())
        pygame.time.wait(sound_length_in_ms_2 + 100)
        self.assertIsNone(channel.get_queue())

    def test_get_sound(self):
        """Ensure Channel.get_sound() returns the currently playing Sound."""
        channel = mixer.Channel(0)
        sound = mixer.Sound(example_path('data/house_lo.wav'))
        channel.play(sound)
        got_sound = channel.get_sound()
        self.assertEqual(got_sound, sound)
        channel.stop()
        got_sound = channel.get_sound()
        self.assertIsNone(got_sound)

    def test_get_volume(self):
        """Ensure a channel's volume can be retrieved."""
        expected_volume = 1.0
        channel = mixer.Channel(0)
        volume = channel.get_volume()
        self.assertAlmostEqual(volume, expected_volume)

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

    def test_pause_unpause__before_init(self):
        """
        Ensure exception for Channel.pause() with non-init mixer.
        """
        sound = mixer.Sound(example_path('data/house_lo.wav'))
        channel = sound.play()
        mixer.quit()
        with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
            channel.pause()
        with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
            channel.unpause()

    def todo_test_queue(self):
        self.fail()

    def test_stop(self):
        channel = mixer.Channel(0)
        sound = mixer.Sound(example_path('data/house_lo.wav'))
        channel.play(sound)
        channel.stop()
        self.assertFalse(channel.get_busy())
        channel.queue(sound)
        channel.stop()
        self.assertFalse(channel.get_busy())
        channel.play(sound)
        self.assertTrue(channel.get_busy())