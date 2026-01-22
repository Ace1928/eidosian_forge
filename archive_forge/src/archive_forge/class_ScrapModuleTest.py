import os
import sys
import unittest
from pygame.tests.test_utils import trunk_relative_path
import pygame
from pygame import scrap
class ScrapModuleTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pygame.display.init()
        pygame.display.set_mode((1, 1))
        scrap.init()

    @classmethod
    def tearDownClass(cls):
        pygame.display.quit()

    def test_init(self):
        """Ensures scrap module still initialized after multiple init calls."""
        scrap.init()
        scrap.init()
        self.assertTrue(scrap.get_init())

    def test_init__reinit(self):
        """Ensures reinitializing the scrap module doesn't clear its data."""
        data_type = pygame.SCRAP_TEXT
        expected_data = b'test_init__reinit'
        scrap.put(data_type, expected_data)
        scrap.init()
        self.assertEqual(scrap.get(data_type), expected_data)

    def test_get_init(self):
        """Ensures get_init gets the init state."""
        self.assertTrue(scrap.get_init())

    def todo_test_contains(self):
        """Ensures contains works as expected."""
        self.fail()

    def todo_test_get(self):
        """Ensures get works as expected."""
        self.fail()

    def test_get__owned_empty_type(self):
        """Ensures get works when there is no data of the requested type
        in the clipboard and the clipboard is owned by the pygame application.
        """
        DATA_TYPE = 'test_get__owned_empty_type'
        if scrap.lost():
            scrap.put(pygame.SCRAP_TEXT, b'text to clipboard')
            if scrap.lost():
                self.skipTest('requires the pygame application to own the clipboard')
        data = scrap.get(DATA_TYPE)
        self.assertIsNone(data)

    def todo_test_get_types(self):
        """Ensures get_types works as expected."""
        self.fail()

    def todo_test_lost(self):
        """Ensures lost works as expected."""
        self.fail()

    def test_set_mode(self):
        """Ensures set_mode works as expected."""
        scrap.set_mode(pygame.SCRAP_SELECTION)
        scrap.set_mode(pygame.SCRAP_CLIPBOARD)
        self.assertRaises(ValueError, scrap.set_mode, 1099)

    def test_put__text(self):
        """Ensures put can place text into the clipboard."""
        scrap.put(pygame.SCRAP_TEXT, b'Hello world')
        self.assertEqual(scrap.get(pygame.SCRAP_TEXT), b'Hello world')
        scrap.put(pygame.SCRAP_TEXT, b'Another String')
        self.assertEqual(scrap.get(pygame.SCRAP_TEXT), b'Another String')

    @unittest.skipIf('pygame.image' not in sys.modules, 'requires pygame.image module')
    def test_put__bmp_image(self):
        """Ensures put can place a BMP image into the clipboard."""
        sf = pygame.image.load(trunk_relative_path('examples/data/asprite.bmp'))
        expected_string = pygame.image.tostring(sf, 'RGBA')
        scrap.put(pygame.SCRAP_BMP, expected_string)
        self.assertEqual(scrap.get(pygame.SCRAP_BMP), expected_string)

    def test_put(self):
        """Ensures put can place data into the clipboard
        when using a user defined type identifier.
        """
        DATA_TYPE = 'arbitrary buffer'
        scrap.put(DATA_TYPE, b'buf')
        r = scrap.get(DATA_TYPE)
        self.assertEqual(r, b'buf')