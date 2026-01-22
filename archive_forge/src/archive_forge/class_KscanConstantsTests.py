import unittest
import pygame.constants
class KscanConstantsTests(unittest.TestCase):
    """Test KSCAN_* (scancode) constants."""
    KSCAN_SPECIFIC_NAMES = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'APOSTROPHE', 'GRAVE', 'INTERNATIONAL1', 'INTERNATIONAL2', 'INTERNATIONAL3', 'INTERNATIONAL4', 'INTERNATIONAL5', 'INTERNATIONAL6', 'INTERNATIONAL7', 'INTERNATIONAL8', 'INTERNATIONAL9', 'LANG1', 'LANG2', 'LANG3', 'LANG4', 'LANG5', 'LANG6', 'LANG7', 'LANG8', 'LANG9', 'NONUSBACKSLASH', 'NONUSHASH')
    KSCAN_NAMES = tuple(('KSCAN_' + n for n in K_AND_KSCAN_COMMON_NAMES + KSCAN_SPECIFIC_NAMES))

    def test_kscan__existence(self):
        """Ensures KSCAN constants exist."""
        for name in self.KSCAN_NAMES:
            self.assertTrue(hasattr(pygame.constants, name), f'missing constant {name}')

    def test_kscan__type(self):
        """Ensures KSCAN constants are the correct type."""
        for name in self.KSCAN_NAMES:
            value = getattr(pygame.constants, name)
            self.assertIs(type(value), int)

    def test_kscan__value_overlap(self):
        """Ensures no unexpected KSCAN constant values overlap."""
        EXPECTED_OVERLAPS = {frozenset(('KSCAN_' + n for n in item)) for item in K_AND_KSCAN_COMMON_OVERLAPS}
        overlaps = create_overlap_set(self.KSCAN_NAMES)
        self.assertSetEqual(overlaps, EXPECTED_OVERLAPS)