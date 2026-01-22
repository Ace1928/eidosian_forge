import unittest
import pygame
class MidiModuleTest(unittest.TestCase):
    """Midi module tests that require midi hardware or midi.init().

    See MidiModuleNonInteractiveTest for non-interactive module tests.
    """
    __tags__ = ['interactive']

    def setUp(self):
        import pygame.midi
        pygame.midi.init()

    def tearDown(self):
        pygame.midi.quit()

    def test_get_count(self):
        c = pygame.midi.get_count()
        self.assertIsInstance(c, int)
        self.assertTrue(c >= 0)

    def test_get_default_input_id(self):
        midin_id = pygame.midi.get_default_input_id()
        self.assertIsInstance(midin_id, int)
        self.assertTrue(midin_id >= -1)
        pygame.midi.quit()
        self.assertRaises(RuntimeError, pygame.midi.get_default_output_id)

    def test_get_default_output_id(self):
        c = pygame.midi.get_default_output_id()
        self.assertIsInstance(c, int)
        self.assertTrue(c >= -1)
        pygame.midi.quit()
        self.assertRaises(RuntimeError, pygame.midi.get_default_output_id)

    def test_get_device_info(self):
        an_id = pygame.midi.get_default_output_id()
        if an_id != -1:
            interf, name, input, output, opened = pygame.midi.get_device_info(an_id)
            self.assertEqual(output, 1)
            self.assertEqual(input, 0)
            self.assertEqual(opened, 0)
        an_in_id = pygame.midi.get_default_input_id()
        if an_in_id != -1:
            r = pygame.midi.get_device_info(an_in_id)
            interf, name, input, output, opened = r
            self.assertEqual(output, 0)
            self.assertEqual(input, 1)
            self.assertEqual(opened, 0)
        out_of_range = pygame.midi.get_count()
        for num in range(out_of_range):
            self.assertIsNotNone(pygame.midi.get_device_info(num))
        info = pygame.midi.get_device_info(out_of_range)
        self.assertIsNone(info)

    def test_init(self):
        pygame.midi.quit()
        self.assertRaises(RuntimeError, pygame.midi.get_count)
        pygame.midi.init()
        pygame.midi.init()
        pygame.midi.init()
        pygame.midi.init()
        self.assertTrue(pygame.midi.get_init())

    def test_quit(self):
        pygame.midi.quit()
        pygame.midi.init()
        pygame.midi.quit()
        pygame.midi.quit()
        pygame.midi.init()
        pygame.midi.init()
        pygame.midi.quit()
        self.assertFalse(pygame.midi.get_init())

    def test_get_init(self):
        self.assertTrue(pygame.midi.get_init())

    def test_time(self):
        mtime = pygame.midi.time()
        self.assertIsInstance(mtime, int)
        self.assertTrue(0 <= mtime < 100)