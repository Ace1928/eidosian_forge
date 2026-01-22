import unittest
import pygame.constants
class KmodConstantsTests(unittest.TestCase):
    """Test KMOD_* (key modifier) constants."""
    KMOD_CONSTANTS = ('KMOD_NONE', 'KMOD_LSHIFT', 'KMOD_RSHIFT', 'KMOD_SHIFT', 'KMOD_LCTRL', 'KMOD_RCTRL', 'KMOD_CTRL', 'KMOD_LALT', 'KMOD_RALT', 'KMOD_ALT', 'KMOD_LMETA', 'KMOD_RMETA', 'KMOD_META', 'KMOD_NUM', 'KMOD_CAPS', 'KMOD_MODE', 'KMOD_LGUI', 'KMOD_RGUI', 'KMOD_GUI')

    def test_kmod__existence(self):
        """Ensures KMOD constants exist."""
        for name in self.KMOD_CONSTANTS:
            self.assertTrue(hasattr(pygame.constants, name), f'missing constant {name}')

    def test_kmod__type(self):
        """Ensures KMOD constants are the correct type."""
        for name in self.KMOD_CONSTANTS:
            value = getattr(pygame.constants, name)
            self.assertIs(type(value), int)

    def test_kmod__value_overlap(self):
        """Ensures no unexpected KMOD constant values overlap."""
        EXPECTED_OVERLAPS = {frozenset(['KMOD_LGUI', 'KMOD_LMETA']), frozenset(['KMOD_RGUI', 'KMOD_RMETA']), frozenset(['KMOD_GUI', 'KMOD_META'])}
        overlaps = create_overlap_set(self.KMOD_CONSTANTS)
        self.assertSetEqual(overlaps, EXPECTED_OVERLAPS)

    def test_kmod__no_bitwise_overlap(self):
        """Ensures certain KMOD constants have no overlapping bits."""
        NO_BITWISE_OVERLAP = ('KMOD_NONE', 'KMOD_LSHIFT', 'KMOD_RSHIFT', 'KMOD_LCTRL', 'KMOD_RCTRL', 'KMOD_LALT', 'KMOD_RALT', 'KMOD_LMETA', 'KMOD_RMETA', 'KMOD_NUM', 'KMOD_CAPS', 'KMOD_MODE')
        kmods = 0
        for name in NO_BITWISE_OVERLAP:
            value = getattr(pygame.constants, name)
            self.assertFalse(kmods & value)
            kmods |= value

    def test_kmod__bitwise_overlap(self):
        """Ensures certain KMOD constants have overlapping bits."""
        KMOD_COMPRISED_DICT = {'KMOD_SHIFT': ('KMOD_LSHIFT', 'KMOD_RSHIFT'), 'KMOD_CTRL': ('KMOD_LCTRL', 'KMOD_RCTRL'), 'KMOD_ALT': ('KMOD_LALT', 'KMOD_RALT'), 'KMOD_META': ('KMOD_LMETA', 'KMOD_RMETA'), 'KMOD_GUI': ('KMOD_LGUI', 'KMOD_RGUI')}
        for base_name, seq_names in KMOD_COMPRISED_DICT.items():
            expected_value = 0
            for name in seq_names:
                expected_value |= getattr(pygame.constants, name)
            value = getattr(pygame.constants, base_name)
            self.assertEqual(value, expected_value)