import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
class SoundTypeTest(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        mixer.quit()

    def setUp(cls):
        if mixer.get_init() is None:
            mixer.init()

    def test_sound(self):
        """Ensure Sound() creation with a filename works."""
        filename = example_path(os.path.join('data', 'house_lo.wav'))
        sound1 = mixer.Sound(filename)
        sound2 = mixer.Sound(file=filename)
        self.assertIsInstance(sound1, mixer.Sound)
        self.assertIsInstance(sound2, mixer.Sound)

    def test_sound__from_file_object(self):
        """Ensure Sound() creation with a file object works."""
        filename = example_path(os.path.join('data', 'house_lo.wav'))
        with open(filename, 'rb') as file_obj:
            sound = mixer.Sound(file_obj)
            self.assertIsInstance(sound, mixer.Sound)

    def test_sound__from_sound_object(self):
        """Ensure Sound() creation with a Sound() object works."""
        filename = example_path(os.path.join('data', 'house_lo.wav'))
        sound_obj = mixer.Sound(file=filename)
        sound = mixer.Sound(sound_obj)
        self.assertIsInstance(sound, mixer.Sound)

    def test_sound__from_pathlib(self):
        """Ensure Sound() creation with a pathlib.Path object works."""
        path = pathlib.Path(example_path(os.path.join('data', 'house_lo.wav')))
        sound1 = mixer.Sound(path)
        sound2 = mixer.Sound(file=path)
        self.assertIsInstance(sound1, mixer.Sound)
        self.assertIsInstance(sound2, mixer.Sound)

    def todo_test_sound__from_buffer(self):
        """Ensure Sound() creation with a buffer works."""
        self.fail()

    def todo_test_sound__from_array(self):
        """Ensure Sound() creation with an array works."""
        self.fail()

    def test_sound__without_arg(self):
        """Ensure exception raised for Sound() creation with no argument."""
        with self.assertRaises(TypeError):
            mixer.Sound()

    def test_sound__before_init(self):
        """Ensure exception raised for Sound() creation with non-init mixer."""
        mixer.quit()
        filename = example_path(os.path.join('data', 'house_lo.wav'))
        with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
            mixer.Sound(file=filename)

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

    def test_get_length(self):
        """Tests if get_length returns a correct length."""
        try:
            for size in SIZES:
                pygame.mixer.quit()
                pygame.mixer.init(size=size)
                filename = example_path(os.path.join('data', 'punch.wav'))
                sound = mixer.Sound(file=filename)
                sound_bytes = sound.get_raw()
                mix_freq, mix_bits, mix_channels = pygame.mixer.get_init()
                mix_bytes = abs(mix_bits) / 8
                expected_length = float(len(sound_bytes)) / mix_freq / mix_bytes / mix_channels
                self.assertAlmostEqual(expected_length, sound.get_length())
        finally:
            pygame.mixer.quit()
            with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
                sound.get_length()

    def test_get_num_channels(self):
        """
        Tests if Sound.get_num_channels returns the correct number
        of channels playing a specific sound.
        """
        try:
            filename = example_path(os.path.join('data', 'house_lo.wav'))
            sound = mixer.Sound(file=filename)
            self.assertEqual(sound.get_num_channels(), 0)
            sound.play()
            self.assertEqual(sound.get_num_channels(), 1)
            sound.play()
            self.assertEqual(sound.get_num_channels(), 2)
            sound.stop()
            self.assertEqual(sound.get_num_channels(), 0)
        finally:
            pygame.mixer.quit()
            with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
                sound.get_num_channels()

    def test_get_volume(self):
        """Ensure a sound's volume can be retrieved."""
        try:
            expected_volume = 1.0
            filename = example_path(os.path.join('data', 'house_lo.wav'))
            sound = mixer.Sound(file=filename)
            volume = sound.get_volume()
            self.assertAlmostEqual(volume, expected_volume)
        finally:
            pygame.mixer.quit()
            with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
                sound.get_volume()

    def test_get_volume__while_playing(self):
        """Ensure a sound's volume can be retrieved while playing."""
        try:
            expected_volume = 1.0
            filename = example_path(os.path.join('data', 'house_lo.wav'))
            sound = mixer.Sound(file=filename)
            sound.play(-1)
            volume = sound.get_volume()
            self.assertAlmostEqual(volume, expected_volume)
        finally:
            pygame.mixer.quit()
            with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
                sound.get_volume()

    def test_set_volume(self):
        """Ensure a sound's volume can be set."""
        try:
            float_delta = 1.0 / 128
            filename = example_path(os.path.join('data', 'house_lo.wav'))
            sound = mixer.Sound(file=filename)
            current_volume = sound.get_volume()
            volumes = ((-1, current_volume), (0, 0.0), (0.01, 0.01), (0.1, 0.1), (0.5, 0.5), (0.9, 0.9), (0.99, 0.99), (1, 1.0), (1.1, 1.0), (2.0, 1.0))
            for volume_set_value, expected_volume in volumes:
                sound.set_volume(volume_set_value)
                self.assertAlmostEqual(sound.get_volume(), expected_volume, delta=float_delta)
        finally:
            pygame.mixer.quit()
            with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
                sound.set_volume(1)

    def test_set_volume__while_playing(self):
        """Ensure a sound's volume can be set while playing."""
        try:
            float_delta = 1.0 / 128
            filename = example_path(os.path.join('data', 'house_lo.wav'))
            sound = mixer.Sound(file=filename)
            current_volume = sound.get_volume()
            volumes = ((-1, current_volume), (0, 0.0), (0.01, 0.01), (0.1, 0.1), (0.5, 0.5), (0.9, 0.9), (0.99, 0.99), (1, 1.0), (1.1, 1.0), (2.0, 1.0))
            sound.play(loops=-1)
            for volume_set_value, expected_volume in volumes:
                sound.set_volume(volume_set_value)
                self.assertAlmostEqual(sound.get_volume(), expected_volume, delta=float_delta)
        finally:
            pygame.mixer.quit()
            with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
                sound.set_volume(1)

    def test_stop(self):
        """Ensure stop can be called while not playing a sound."""
        try:
            expected_channels = 0
            filename = example_path(os.path.join('data', 'house_lo.wav'))
            sound = mixer.Sound(file=filename)
            sound.stop()
            self.assertEqual(sound.get_num_channels(), expected_channels)
        finally:
            pygame.mixer.quit()
            with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
                sound.stop()

    def test_stop__while_playing(self):
        """Ensure stop stops a playing sound."""
        try:
            expected_channels = 0
            filename = example_path(os.path.join('data', 'house_lo.wav'))
            sound = mixer.Sound(file=filename)
            sound.play(-1)
            sound.stop()
            self.assertEqual(sound.get_num_channels(), expected_channels)
        finally:
            pygame.mixer.quit()
            with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
                sound.stop()

    def test_get_raw(self):
        """Ensure get_raw returns the correct bytestring."""
        try:
            samples = b'abcdefgh'
            snd = mixer.Sound(buffer=samples)
            raw = snd.get_raw()
            self.assertIsInstance(raw, bytes)
            self.assertEqual(raw, samples)
        finally:
            pygame.mixer.quit()
            with self.assertRaisesRegex(pygame.error, 'mixer not initialized'):
                snd.get_raw()

    def test_correct_subclassing(self):

        class CorrectSublass(mixer.Sound):

            def __init__(self, file):
                super().__init__(file=file)
        filename = example_path(os.path.join('data', 'house_lo.wav'))
        correct = CorrectSublass(filename)
        try:
            correct.get_volume()
        except Exception:
            self.fail('This should not raise an exception.')

    def test_incorrect_subclassing(self):

        class IncorrectSuclass(mixer.Sound):

            def __init__(self):
                pass
        incorrect = IncorrectSuclass()
        self.assertRaises(RuntimeError, incorrect.get_volume)