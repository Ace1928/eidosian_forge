from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
class MaskTypeTest(unittest.TestCase):
    ORIGIN_OFFSETS = ((0, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))

    @unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
    def test_mask(self):
        """Ensure masks are created correctly without fill parameter."""
        expected_count = 0
        expected_size = (11, 23)
        mask1 = pygame.mask.Mask(expected_size)
        mask2 = pygame.mask.Mask(size=expected_size)
        self.assertIsInstance(mask1, pygame.mask.Mask)
        self.assertEqual(mask1.count(), expected_count)
        self.assertEqual(mask1.get_size(), expected_size)
        self.assertIsInstance(mask2, pygame.mask.Mask)
        self.assertEqual(mask2.count(), expected_count)
        self.assertEqual(mask2.get_size(), expected_size)

    def test_mask__negative_size(self):
        """Ensure the mask constructor handles negative sizes correctly."""
        for size in ((1, -1), (-1, 1), (-1, -1)):
            with self.assertRaises(ValueError):
                mask = pygame.Mask(size)

    def test_mask__fill_kwarg(self):
        """Ensure masks are created correctly using the fill keyword."""
        width, height = (37, 47)
        expected_size = (width, height)
        fill_counts = {True: width * height, False: 0}
        for fill, expected_count in fill_counts.items():
            msg = f'fill={fill}'
            mask = pygame.mask.Mask(expected_size, fill=fill)
            self.assertIsInstance(mask, pygame.mask.Mask, msg)
            self.assertEqual(mask.count(), expected_count, msg)
            self.assertEqual(mask.get_size(), expected_size, msg)

    @unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
    def test_mask__fill_kwarg_bit_boundaries(self):
        """Ensures masks are created correctly using the fill keyword
        over a range of sizes.

        Tests masks of different sizes, including:
           -masks 31 to 33 bits wide (32 bit boundaries)
           -masks 63 to 65 bits wide (64 bit boundaries)
        """
        for height in range(1, 4):
            for width in range(1, 66):
                expected_count = width * height
                expected_size = (width, height)
                msg = f'size={expected_size}'
                mask = pygame.mask.Mask(expected_size, fill=True)
                self.assertIsInstance(mask, pygame.mask.Mask, msg)
                self.assertEqual(mask.count(), expected_count, msg)
                self.assertEqual(mask.get_size(), expected_size, msg)

    def test_mask__fill_arg(self):
        """Ensure masks are created correctly using a fill arg."""
        width, height = (59, 71)
        expected_size = (width, height)
        fill_counts = {True: width * height, False: 0}
        for fill, expected_count in fill_counts.items():
            msg = f'fill={fill}'
            mask = pygame.mask.Mask(expected_size, fill)
            self.assertIsInstance(mask, pygame.mask.Mask, msg)
            self.assertEqual(mask.count(), expected_count, msg)
            self.assertEqual(mask.get_size(), expected_size, msg)

    def test_mask__size_kwarg(self):
        """Ensure masks are created correctly using the size keyword."""
        width, height = (73, 83)
        expected_size = (width, height)
        fill_counts = {True: width * height, False: 0}
        for fill, expected_count in fill_counts.items():
            msg = f'fill={fill}'
            mask1 = pygame.mask.Mask(fill=fill, size=expected_size)
            mask2 = pygame.mask.Mask(size=expected_size, fill=fill)
            self.assertIsInstance(mask1, pygame.mask.Mask, msg)
            self.assertIsInstance(mask2, pygame.mask.Mask, msg)
            self.assertEqual(mask1.count(), expected_count, msg)
            self.assertEqual(mask2.count(), expected_count, msg)
            self.assertEqual(mask1.get_size(), expected_size, msg)
            self.assertEqual(mask2.get_size(), expected_size, msg)

    def test_copy(self):
        """Ensures copy works correctly with some bits set and unset."""
        for width in (31, 32, 33, 63, 64, 65):
            for height in (31, 32, 33, 63, 64, 65):
                mask = pygame.mask.Mask((width, height))
                for x in range(width):
                    for y in range(x & 1, height, 2):
                        mask.set_at((x, y))
                for mask_copy in (mask.copy(), copy.copy(mask)):
                    self.assertIsInstance(mask_copy, pygame.mask.Mask)
                    self.assertIsNot(mask_copy, mask)
                    assertMaskEqual(self, mask_copy, mask)

    def test_copy__full(self):
        """Ensures copy works correctly on a filled masked."""
        for width in (31, 32, 33, 63, 64, 65):
            for height in (31, 32, 33, 63, 64, 65):
                mask = pygame.mask.Mask((width, height), fill=True)
                for mask_copy in (mask.copy(), copy.copy(mask)):
                    self.assertIsInstance(mask_copy, pygame.mask.Mask)
                    self.assertIsNot(mask_copy, mask)
                    assertMaskEqual(self, mask_copy, mask)

    def test_copy__empty(self):
        """Ensures copy works correctly on an empty mask."""
        for width in (31, 32, 33, 63, 64, 65):
            for height in (31, 32, 33, 63, 64, 65):
                mask = pygame.mask.Mask((width, height))
                for mask_copy in (mask.copy(), copy.copy(mask)):
                    self.assertIsInstance(mask_copy, pygame.mask.Mask)
                    self.assertIsNot(mask_copy, mask)
                    assertMaskEqual(self, mask_copy, mask)

    def test_copy__independent(self):
        """Ensures copy makes an independent copy of the mask."""
        mask_set_pos = (64, 1)
        mask_copy_set_pos = (64, 2)
        mask = pygame.mask.Mask((65, 3))
        mask_copies = (mask.copy(), copy.copy(mask))
        mask.set_at(mask_set_pos)
        for mask_copy in mask_copies:
            mask_copy.set_at(mask_copy_set_pos)
            self.assertIsNot(mask_copy, mask)
            self.assertNotEqual(mask_copy.get_at(mask_set_pos), mask.get_at(mask_set_pos))
            self.assertNotEqual(mask_copy.get_at(mask_copy_set_pos), mask.get_at(mask_copy_set_pos))

    def test_get_size(self):
        """Ensure a mask's size is correctly retrieved."""
        expected_size = (93, 101)
        mask = pygame.mask.Mask(expected_size)
        self.assertEqual(mask.get_size(), expected_size)

    def test_get_rect(self):
        """Ensures get_rect works correctly."""
        expected_rect = pygame.Rect((0, 0), (11, 13))
        for fill in (True, False):
            mask = pygame.mask.Mask(expected_rect.size, fill=fill)
            rect = mask.get_rect()
            self.assertEqual(rect, expected_rect)

    @unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
    def test_get_rect__one_kwarg(self):
        """Ensures get_rect supports a single rect attribute kwarg.

        Tests all the rect attributes.
        """
        RECT_SINGLE_VALUE_ATTRIBUTES = ('x', 'y', 'top', 'left', 'bottom', 'right', 'centerx', 'centery', 'width', 'height', 'w', 'h')
        RECT_DOUBLE_VALUE_ATTRIBUTES = ('topleft', 'bottomleft', 'topright', 'bottomright', 'midtop', 'midleft', 'midbottom', 'midright', 'center', 'size')
        rect_attributes = {RECT_SINGLE_VALUE_ATTRIBUTES: (3, 5.1), RECT_DOUBLE_VALUE_ATTRIBUTES: ((1, 2.2), [2.3, 3], Vector2(0, 1))}
        size = (7, 3)
        mask = pygame.mask.Mask(size)
        for attributes, values in rect_attributes.items():
            for attribute in attributes:
                for value in values:
                    expected_rect = pygame.Rect((0, 0), size)
                    setattr(expected_rect, attribute, value)
                    rect = mask.get_rect(**{attribute: value})
                    self.assertEqual(rect, expected_rect)

    def test_get_rect__multiple_kwargs(self):
        """Ensures get_rect supports multiple rect attribute kwargs."""
        mask = pygame.mask.Mask((5, 4))
        expected_rect = pygame.Rect((0, 0), (0, 0))
        kwargs = {'x': 7.1, 'top': -1, 'size': Vector2(2, 3.2)}
        for attrib, value in kwargs.items():
            setattr(expected_rect, attrib, value)
        rect = mask.get_rect(**kwargs)
        self.assertEqual(rect, expected_rect)

    def test_get_rect__no_arg_support(self):
        """Ensures get_rect only supports kwargs."""
        mask = pygame.mask.Mask((4, 5))
        with self.assertRaises(TypeError):
            rect = mask.get_rect(3)
        with self.assertRaises(TypeError):
            rect = mask.get_rect((1, 2))

    @unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
    def test_get_rect__invalid_kwarg_name(self):
        """Ensures get_rect detects invalid kwargs."""
        mask = pygame.mask.Mask((1, 2))
        with self.assertRaises(AttributeError):
            rect = mask.get_rect(righte=11)
        with self.assertRaises(AttributeError):
            rect = mask.get_rect(toplef=(1, 1))
        with self.assertRaises(AttributeError):
            rect = mask.get_rect(move=(3, 2))

    def test_get_rect__invalid_kwarg_format(self):
        """Ensures get_rect detects invalid kwarg formats."""
        mask = pygame.mask.Mask((3, 11))
        with self.assertRaises(TypeError):
            rect = mask.get_rect(right='1')
        with self.assertRaises(TypeError):
            rect = mask.get_rect(bottom=(1,))
        with self.assertRaises(TypeError):
            rect = mask.get_rect(centerx=(1, 1))
        with self.assertRaises(TypeError):
            rect = mask.get_rect(midleft=(1, '1'))
        with self.assertRaises(TypeError):
            rect = mask.get_rect(topright=(1,))
        with self.assertRaises(TypeError):
            rect = mask.get_rect(bottomleft=(1, 2, 3))
        with self.assertRaises(TypeError):
            rect = mask.get_rect(midbottom=1)

    def test_get_at(self):
        """Ensure individual mask bits are correctly retrieved."""
        width, height = (5, 7)
        mask0 = pygame.mask.Mask((width, height))
        mask1 = pygame.mask.Mask((width, height), fill=True)
        mask0_expected_bit = 0
        mask1_expected_bit = 1
        pos = (width - 1, height - 1)
        self.assertEqual(mask0.get_at(pos), mask0_expected_bit)
        self.assertEqual(mask0.get_at(pos=pos), mask0_expected_bit)
        self.assertEqual(mask1.get_at(Vector2(pos)), mask1_expected_bit)
        self.assertEqual(mask1.get_at(pos=Vector2(pos)), mask1_expected_bit)

    def test_get_at__out_of_bounds(self):
        """Ensure get_at() checks bounds."""
        width, height = (11, 3)
        mask = pygame.mask.Mask((width, height))
        with self.assertRaises(IndexError):
            mask.get_at((width, 0))
        with self.assertRaises(IndexError):
            mask.get_at((0, height))
        with self.assertRaises(IndexError):
            mask.get_at((-1, 0))
        with self.assertRaises(IndexError):
            mask.get_at((0, -1))

    def test_set_at(self):
        """Ensure individual mask bits are set to 1."""
        width, height = (13, 17)
        mask0 = pygame.mask.Mask((width, height))
        mask1 = pygame.mask.Mask((width, height), fill=True)
        mask0_expected_count = 1
        mask1_expected_count = mask1.count()
        expected_bit = 1
        pos = (width - 1, height - 1)
        mask0.set_at(pos, expected_bit)
        mask1.set_at(pos=Vector2(pos), value=expected_bit)
        self.assertEqual(mask0.get_at(pos), expected_bit)
        self.assertEqual(mask0.count(), mask0_expected_count)
        self.assertEqual(mask1.get_at(pos), expected_bit)
        self.assertEqual(mask1.count(), mask1_expected_count)

    def test_set_at__to_0(self):
        """Ensure individual mask bits are set to 0."""
        width, height = (11, 7)
        mask0 = pygame.mask.Mask((width, height))
        mask1 = pygame.mask.Mask((width, height), fill=True)
        mask0_expected_count = 0
        mask1_expected_count = mask1.count() - 1
        expected_bit = 0
        pos = (width - 1, height - 1)
        mask0.set_at(pos, expected_bit)
        mask1.set_at(pos, expected_bit)
        self.assertEqual(mask0.get_at(pos), expected_bit)
        self.assertEqual(mask0.count(), mask0_expected_count)
        self.assertEqual(mask1.get_at(pos), expected_bit)
        self.assertEqual(mask1.count(), mask1_expected_count)

    def test_set_at__default_value(self):
        """Ensure individual mask bits are set using the default value."""
        width, height = (3, 21)
        mask0 = pygame.mask.Mask((width, height))
        mask1 = pygame.mask.Mask((width, height), fill=True)
        mask0_expected_count = 1
        mask1_expected_count = mask1.count()
        expected_bit = 1
        pos = (width - 1, height - 1)
        mask0.set_at(pos)
        mask1.set_at(pos)
        self.assertEqual(mask0.get_at(pos), expected_bit)
        self.assertEqual(mask0.count(), mask0_expected_count)
        self.assertEqual(mask1.get_at(pos), expected_bit)
        self.assertEqual(mask1.count(), mask1_expected_count)

    def test_set_at__out_of_bounds(self):
        """Ensure set_at() checks bounds."""
        width, height = (11, 3)
        mask = pygame.mask.Mask((width, height))
        with self.assertRaises(IndexError):
            mask.set_at((width, 0))
        with self.assertRaises(IndexError):
            mask.set_at((0, height))
        with self.assertRaises(IndexError):
            mask.set_at((-1, 0))
        with self.assertRaises(IndexError):
            mask.set_at((0, -1))

    def test_overlap(self):
        """Ensure the overlap intersection is correctly calculated.

        Testing the different combinations of full/empty masks:
            (mask1-filled) 1 overlap 1 (mask2-filled)
            (mask1-empty)  0 overlap 1 (mask2-filled)
            (mask1-filled) 1 overlap 0 (mask2-empty)
            (mask1-empty)  0 overlap 0 (mask2-empty)
        """
        expected_size = (4, 4)
        offset = (0, 0)
        expected_default = None
        expected_overlaps = {(True, True): offset}
        for fill2 in (True, False):
            mask2 = pygame.mask.Mask(expected_size, fill=fill2)
            mask2_count = mask2.count()
            for fill1 in (True, False):
                key = (fill1, fill2)
                msg = f'key={key}'
                mask1 = pygame.mask.Mask(expected_size, fill=fill1)
                mask1_count = mask1.count()
                expected_pos = expected_overlaps.get(key, expected_default)
                overlap_pos = mask1.overlap(mask2, offset)
                self.assertEqual(overlap_pos, expected_pos, msg)
                self.assertEqual(mask1.count(), mask1_count, msg)
                self.assertEqual(mask2.count(), mask2_count, msg)
                self.assertEqual(mask1.get_size(), expected_size, msg)
                self.assertEqual(mask2.get_size(), expected_size, msg)

    def test_overlap__offset(self):
        """Ensure an offset overlap intersection is correctly calculated."""
        mask1 = pygame.mask.Mask((65, 3), fill=True)
        mask2 = pygame.mask.Mask((66, 4), fill=True)
        mask1_count = mask1.count()
        mask2_count = mask2.count()
        mask1_size = mask1.get_size()
        mask2_size = mask2.get_size()
        for offset in self.ORIGIN_OFFSETS:
            msg = f'offset={offset}'
            expected_pos = (max(offset[0], 0), max(offset[1], 0))
            overlap_pos = mask1.overlap(other=mask2, offset=offset)
            self.assertEqual(overlap_pos, expected_pos, msg)
            self.assertEqual(mask1.count(), mask1_count, msg)
            self.assertEqual(mask2.count(), mask2_count, msg)
            self.assertEqual(mask1.get_size(), mask1_size, msg)
            self.assertEqual(mask2.get_size(), mask2_size, msg)

    def test_overlap__offset_with_unset_bits(self):
        """Ensure an offset overlap intersection is correctly calculated
        when (0, 0) bits not set."""
        mask1 = pygame.mask.Mask((65, 3), fill=True)
        mask2 = pygame.mask.Mask((66, 4), fill=True)
        unset_pos = (0, 0)
        mask1.set_at(unset_pos, 0)
        mask2.set_at(unset_pos, 0)
        mask1_count = mask1.count()
        mask2_count = mask2.count()
        mask1_size = mask1.get_size()
        mask2_size = mask2.get_size()
        for offset in self.ORIGIN_OFFSETS:
            msg = f'offset={offset}'
            x, y = offset
            expected_y = max(y, 0)
            if 0 == y:
                expected_x = max(x + 1, 1)
            elif 0 < y:
                expected_x = max(x + 1, 0)
            else:
                expected_x = max(x, 1)
            overlap_pos = mask1.overlap(mask2, Vector2(offset))
            self.assertEqual(overlap_pos, (expected_x, expected_y), msg)
            self.assertEqual(mask1.count(), mask1_count, msg)
            self.assertEqual(mask2.count(), mask2_count, msg)
            self.assertEqual(mask1.get_size(), mask1_size, msg)
            self.assertEqual(mask2.get_size(), mask2_size, msg)
            self.assertEqual(mask1.get_at(unset_pos), 0, msg)
            self.assertEqual(mask2.get_at(unset_pos), 0, msg)

    def test_overlap__no_overlap(self):
        """Ensure an offset overlap intersection is correctly calculated
        when there is no overlap."""
        mask1 = pygame.mask.Mask((65, 3), fill=True)
        mask1_count = mask1.count()
        mask1_size = mask1.get_size()
        mask2_w, mask2_h = (67, 5)
        mask2_size = (mask2_w, mask2_h)
        mask2 = pygame.mask.Mask(mask2_size)
        set_pos = (mask2_w - 1, mask2_h - 1)
        mask2.set_at(set_pos)
        mask2_count = 1
        for offset in self.ORIGIN_OFFSETS:
            msg = f'offset={offset}'
            overlap_pos = mask1.overlap(mask2, offset)
            self.assertIsNone(overlap_pos, msg)
            self.assertEqual(mask1.count(), mask1_count, msg)
            self.assertEqual(mask2.count(), mask2_count, msg)
            self.assertEqual(mask1.get_size(), mask1_size, msg)
            self.assertEqual(mask2.get_size(), mask2_size, msg)
            self.assertEqual(mask2.get_at(set_pos), 1, msg)

    def test_overlap__offset_boundary(self):
        """Ensures overlap handles offsets and boundaries correctly."""
        mask1 = pygame.mask.Mask((13, 3), fill=True)
        mask2 = pygame.mask.Mask((7, 5), fill=True)
        mask1_count = mask1.count()
        mask2_count = mask2.count()
        mask1_size = mask1.get_size()
        mask2_size = mask2.get_size()
        offsets = ((mask1_size[0], 0), (0, mask1_size[1]), (-mask2_size[0], 0), (0, -mask2_size[1]))
        for offset in offsets:
            msg = f'offset={offset}'
            overlap_pos = mask1.overlap(mask2, offset)
            self.assertIsNone(overlap_pos, msg)
            self.assertEqual(mask1.count(), mask1_count, msg)
            self.assertEqual(mask2.count(), mask2_count, msg)
            self.assertEqual(mask1.get_size(), mask1_size, msg)
            self.assertEqual(mask2.get_size(), mask2_size, msg)

    @unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
    def test_overlap__bit_boundaries(self):
        """Ensures overlap handles masks of different sizes correctly.

        Tests masks of different sizes, including:
           -masks 31 to 33 bits wide (32 bit boundaries)
           -masks 63 to 65 bits wide (64 bit boundaries)
        """
        for height in range(2, 4):
            for width in range(2, 66):
                mask_size = (width, height)
                mask_count = width * height
                mask1 = pygame.mask.Mask(mask_size, fill=True)
                mask2 = pygame.mask.Mask(mask_size, fill=True)
                for offset in self.ORIGIN_OFFSETS:
                    msg = f'size={mask_size}, offset={offset}'
                    expected_pos = (max(offset[0], 0), max(offset[1], 0))
                    overlap_pos = mask1.overlap(mask2, offset)
                    self.assertEqual(overlap_pos, expected_pos, msg)
                    self.assertEqual(mask1.count(), mask_count, msg)
                    self.assertEqual(mask2.count(), mask_count, msg)
                    self.assertEqual(mask1.get_size(), mask_size, msg)
                    self.assertEqual(mask2.get_size(), mask_size, msg)

    @unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
    def test_overlap__invalid_mask_arg(self):
        """Ensure overlap handles invalid mask arguments correctly."""
        size = (5, 3)
        offset = (0, 0)
        mask = pygame.mask.Mask(size)
        invalid_mask = pygame.Surface(size)
        with self.assertRaises(TypeError):
            overlap_pos = mask.overlap(invalid_mask, offset)

    def test_overlap__invalid_offset_arg(self):
        """Ensure overlap handles invalid offset arguments correctly."""
        size = (2, 7)
        offset = '(0, 0)'
        mask1 = pygame.mask.Mask(size)
        mask2 = pygame.mask.Mask(size)
        with self.assertRaises(TypeError):
            overlap_pos = mask1.overlap(mask2, offset)

    @unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
    def test_overlap_area(self):
        """Ensure the overlap_area is correctly calculated.

        Testing the different combinations of full/empty masks:
            (mask1-filled) 1 overlap_area 1 (mask2-filled)
            (mask1-empty)  0 overlap_area 1 (mask2-filled)
            (mask1-filled) 1 overlap_area 0 (mask2-empty)
            (mask1-empty)  0 overlap_area 0 (mask2-empty)
        """
        expected_size = width, height = (4, 4)
        offset = (0, 0)
        expected_default = 0
        expected_counts = {(True, True): width * height}
        for fill2 in (True, False):
            mask2 = pygame.mask.Mask(expected_size, fill=fill2)
            mask2_count = mask2.count()
            for fill1 in (True, False):
                key = (fill1, fill2)
                msg = f'key={key}'
                mask1 = pygame.mask.Mask(expected_size, fill=fill1)
                mask1_count = mask1.count()
                expected_count = expected_counts.get(key, expected_default)
                overlap_count = mask1.overlap_area(mask2, offset)
                self.assertEqual(overlap_count, expected_count, msg)
                self.assertEqual(mask1.count(), mask1_count, msg)
                self.assertEqual(mask2.count(), mask2_count, msg)
                self.assertEqual(mask1.get_size(), expected_size, msg)
                self.assertEqual(mask2.get_size(), expected_size, msg)

    @unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
    def test_overlap_area__offset(self):
        """Ensure an offset overlap_area is correctly calculated."""
        mask1 = pygame.mask.Mask((65, 3), fill=True)
        mask2 = pygame.mask.Mask((66, 4), fill=True)
        mask1_count = mask1.count()
        mask2_count = mask2.count()
        mask1_size = mask1.get_size()
        mask2_size = mask2.get_size()
        rect1 = mask1.get_rect()
        rect2 = mask2.get_rect()
        for offset in self.ORIGIN_OFFSETS:
            msg = f'offset={offset}'
            rect2.topleft = offset
            overlap_rect = rect1.clip(rect2)
            expected_count = overlap_rect.w * overlap_rect.h
            overlap_count = mask1.overlap_area(other=mask2, offset=offset)
            self.assertEqual(overlap_count, expected_count, msg)
            self.assertEqual(mask1.count(), mask1_count, msg)
            self.assertEqual(mask2.count(), mask2_count, msg)
            self.assertEqual(mask1.get_size(), mask1_size, msg)
            self.assertEqual(mask2.get_size(), mask2_size, msg)

    def test_overlap_area__offset_boundary(self):
        """Ensures overlap_area handles offsets and boundaries correctly."""
        mask1 = pygame.mask.Mask((11, 3), fill=True)
        mask2 = pygame.mask.Mask((5, 7), fill=True)
        mask1_count = mask1.count()
        mask2_count = mask2.count()
        mask1_size = mask1.get_size()
        mask2_size = mask2.get_size()
        expected_count = 0
        offsets = ((mask1_size[0], 0), (0, mask1_size[1]), (-mask2_size[0], 0), (0, -mask2_size[1]))
        for offset in offsets:
            msg = f'offset={offset}'
            overlap_count = mask1.overlap_area(mask2, Vector2(offset))
            self.assertEqual(overlap_count, expected_count, msg)
            self.assertEqual(mask1.count(), mask1_count, msg)
            self.assertEqual(mask2.count(), mask2_count, msg)
            self.assertEqual(mask1.get_size(), mask1_size, msg)
            self.assertEqual(mask2.get_size(), mask2_size, msg)

    @unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
    def test_overlap_area__bit_boundaries(self):
        """Ensures overlap_area handles masks of different sizes correctly.

        Tests masks of different sizes, including:
           -masks 31 to 33 bits wide (32 bit boundaries)
           -masks 63 to 65 bits wide (64 bit boundaries)
        """
        for height in range(2, 4):
            for width in range(2, 66):
                mask_size = (width, height)
                mask_count = width * height
                mask1 = pygame.mask.Mask(mask_size, fill=True)
                mask2 = pygame.mask.Mask(mask_size, fill=True)
                rect1 = mask1.get_rect()
                rect2 = mask2.get_rect()
                for offset in self.ORIGIN_OFFSETS:
                    msg = f'size={mask_size}, offset={offset}'
                    rect2.topleft = offset
                    overlap_rect = rect1.clip(rect2)
                    expected_overlap_count = overlap_rect.w * overlap_rect.h
                    overlap_count = mask1.overlap_area(mask2, offset)
                    self.assertEqual(overlap_count, expected_overlap_count, msg)
                    self.assertEqual(mask1.count(), mask_count, msg)
                    self.assertEqual(mask2.count(), mask_count, msg)
                    self.assertEqual(mask1.get_size(), mask_size, msg)
                    self.assertEqual(mask2.get_size(), mask_size, msg)

    def test_overlap_area__invalid_mask_arg(self):
        """Ensure overlap_area handles invalid mask arguments correctly."""
        size = (3, 5)
        offset = (0, 0)
        mask = pygame.mask.Mask(size)
        invalid_mask = pygame.Surface(size)
        with self.assertRaises(TypeError):
            overlap_count = mask.overlap_area(invalid_mask, offset)

    def test_overlap_area__invalid_offset_arg(self):
        """Ensure overlap_area handles invalid offset arguments correctly."""
        size = (7, 2)
        offset = '(0, 0)'
        mask1 = pygame.mask.Mask(size)
        mask2 = pygame.mask.Mask(size)
        with self.assertRaises(TypeError):
            overlap_count = mask1.overlap_area(mask2, offset)

    def test_overlap_mask(self):
        """Ensure overlap_mask's mask has correct bits set.

        Testing the different combinations of full/empty masks:
            (mask1-filled) 1 overlap_mask 1 (mask2-filled)
            (mask1-empty)  0 overlap_mask 1 (mask2-filled)
            (mask1-filled) 1 overlap_mask 0 (mask2-empty)
            (mask1-empty)  0 overlap_mask 0 (mask2-empty)
        """
        expected_size = (4, 4)
        offset = (0, 0)
        expected_default = pygame.mask.Mask(expected_size)
        expected_masks = {(True, True): pygame.mask.Mask(expected_size, fill=True)}
        for fill2 in (True, False):
            mask2 = pygame.mask.Mask(expected_size, fill=fill2)
            mask2_count = mask2.count()
            for fill1 in (True, False):
                key = (fill1, fill2)
                msg = f'key={key}'
                mask1 = pygame.mask.Mask(expected_size, fill=fill1)
                mask1_count = mask1.count()
                expected_mask = expected_masks.get(key, expected_default)
                overlap_mask = mask1.overlap_mask(other=mask2, offset=offset)
                self.assertIsInstance(overlap_mask, pygame.mask.Mask, msg)
                assertMaskEqual(self, overlap_mask, expected_mask, msg)
                self.assertEqual(mask1.count(), mask1_count, msg)
                self.assertEqual(mask2.count(), mask2_count, msg)
                self.assertEqual(mask1.get_size(), expected_size, msg)
                self.assertEqual(mask2.get_size(), expected_size, msg)

    @unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
    def test_overlap_mask__bits_set(self):
        """Ensure overlap_mask's mask has correct bits set."""
        mask1 = pygame.mask.Mask((50, 50), fill=True)
        mask2 = pygame.mask.Mask((300, 10), fill=True)
        mask1_count = mask1.count()
        mask2_count = mask2.count()
        mask1_size = mask1.get_size()
        mask2_size = mask2.get_size()
        mask3 = mask1.overlap_mask(mask2, (-1, 0))
        for i in range(50):
            for j in range(10):
                self.assertEqual(mask3.get_at((i, j)), 1, f'({i}, {j})')
        for i in range(50):
            for j in range(11, 50):
                self.assertEqual(mask3.get_at((i, j)), 0, f'({i}, {j})')
        self.assertEqual(mask1.count(), mask1_count)
        self.assertEqual(mask2.count(), mask2_count)
        self.assertEqual(mask1.get_size(), mask1_size)
        self.assertEqual(mask2.get_size(), mask2_size)

    def test_overlap_mask__offset(self):
        """Ensure an offset overlap_mask's mask is correctly calculated."""
        mask1 = pygame.mask.Mask((65, 3), fill=True)
        mask2 = pygame.mask.Mask((66, 4), fill=True)
        mask1_count = mask1.count()
        mask2_count = mask2.count()
        mask1_size = mask1.get_size()
        mask2_size = mask2.get_size()
        expected_mask = pygame.Mask(mask1_size)
        rect1 = mask1.get_rect()
        rect2 = mask2.get_rect()
        for offset in self.ORIGIN_OFFSETS:
            msg = f'offset={offset}'
            rect2.topleft = offset
            overlap_rect = rect1.clip(rect2)
            expected_mask.clear()
            expected_mask.draw(pygame.Mask(overlap_rect.size, fill=True), overlap_rect.topleft)
            overlap_mask = mask1.overlap_mask(mask2, offset)
            self.assertIsInstance(overlap_mask, pygame.mask.Mask, msg)
            assertMaskEqual(self, overlap_mask, expected_mask, msg)
            self.assertEqual(mask1.count(), mask1_count, msg)
            self.assertEqual(mask2.count(), mask2_count, msg)
            self.assertEqual(mask1.get_size(), mask1_size, msg)
            self.assertEqual(mask2.get_size(), mask2_size, msg)

    @unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
    def test_overlap_mask__specific_offsets(self):
        """Ensure an offset overlap_mask's mask is correctly calculated.

        Testing the specific case of:
            -both masks are wider than 32 bits
            -a positive offset is used
            -the mask calling overlap_mask() is wider than the mask passed in
        """
        mask1 = pygame.mask.Mask((65, 5), fill=True)
        mask2 = pygame.mask.Mask((33, 3), fill=True)
        expected_mask = pygame.Mask(mask1.get_size())
        rect1 = mask1.get_rect()
        rect2 = mask2.get_rect()
        corner_rect = rect1.inflate(-2, -2)
        for corner in ('topleft', 'topright', 'bottomright', 'bottomleft'):
            setattr(rect2, corner, getattr(corner_rect, corner))
            offset = rect2.topleft
            msg = f'offset={offset}'
            overlap_rect = rect1.clip(rect2)
            expected_mask.clear()
            expected_mask.draw(pygame.Mask(overlap_rect.size, fill=True), overlap_rect.topleft)
            overlap_mask = mask1.overlap_mask(mask2, offset)
            self.assertIsInstance(overlap_mask, pygame.mask.Mask, msg)
            assertMaskEqual(self, overlap_mask, expected_mask, msg)

    def test_overlap_mask__offset_boundary(self):
        """Ensures overlap_mask handles offsets and boundaries correctly."""
        mask1 = pygame.mask.Mask((9, 3), fill=True)
        mask2 = pygame.mask.Mask((11, 5), fill=True)
        mask1_count = mask1.count()
        mask2_count = mask2.count()
        mask1_size = mask1.get_size()
        mask2_size = mask2.get_size()
        expected_count = 0
        expected_size = mask1_size
        offsets = ((mask1_size[0], 0), (0, mask1_size[1]), (-mask2_size[0], 0), (0, -mask2_size[1]))
        for offset in offsets:
            msg = f'offset={offset}'
            overlap_mask = mask1.overlap_mask(mask2, offset)
            self.assertIsInstance(overlap_mask, pygame.mask.Mask, msg)
            self.assertEqual(overlap_mask.count(), expected_count, msg)
            self.assertEqual(overlap_mask.get_size(), expected_size, msg)
            self.assertEqual(mask1.count(), mask1_count, msg)
            self.assertEqual(mask2.count(), mask2_count, msg)
            self.assertEqual(mask1.get_size(), mask1_size, msg)
            self.assertEqual(mask2.get_size(), mask2_size, msg)

    @unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
    def test_overlap_mask__bit_boundaries(self):
        """Ensures overlap_mask handles masks of different sizes correctly.

        Tests masks of different sizes, including:
           -masks 31 to 33 bits wide (32 bit boundaries)
           -masks 63 to 65 bits wide (64 bit boundaries)
        """
        for height in range(2, 4):
            for width in range(2, 66):
                mask_size = (width, height)
                mask_count = width * height
                mask1 = pygame.mask.Mask(mask_size, fill=True)
                mask2 = pygame.mask.Mask(mask_size, fill=True)
                expected_mask = pygame.Mask(mask_size)
                rect1 = mask1.get_rect()
                rect2 = mask2.get_rect()
                for offset in self.ORIGIN_OFFSETS:
                    msg = f'size={mask_size}, offset={offset}'
                    rect2.topleft = offset
                    overlap_rect = rect1.clip(rect2)
                    expected_mask.clear()
                    expected_mask.draw(pygame.Mask(overlap_rect.size, fill=True), overlap_rect.topleft)
                    overlap_mask = mask1.overlap_mask(mask2, offset)
                    self.assertIsInstance(overlap_mask, pygame.mask.Mask, msg)
                    assertMaskEqual(self, overlap_mask, expected_mask, msg)
                    self.assertEqual(mask1.count(), mask_count, msg)
                    self.assertEqual(mask2.count(), mask_count, msg)
                    self.assertEqual(mask1.get_size(), mask_size, msg)
                    self.assertEqual(mask2.get_size(), mask_size, msg)

    def test_overlap_mask__invalid_mask_arg(self):
        """Ensure overlap_mask handles invalid mask arguments correctly."""
        size = (3, 2)
        offset = (0, 0)
        mask = pygame.mask.Mask(size)
        invalid_mask = pygame.Surface(size)
        with self.assertRaises(TypeError):
            overlap_mask = mask.overlap_mask(invalid_mask, offset)

    def test_overlap_mask__invalid_offset_arg(self):
        """Ensure overlap_mask handles invalid offset arguments correctly."""
        size = (5, 2)
        offset = '(0, 0)'
        mask1 = pygame.mask.Mask(size)
        mask2 = pygame.mask.Mask(size)
        with self.assertRaises(TypeError):
            overlap_mask = mask1.overlap_mask(mask2, offset)

    def test_mask_access(self):
        """do the set_at, and get_at parts work correctly?"""
        m = pygame.Mask((10, 10))
        m.set_at((0, 0), 1)
        self.assertEqual(m.get_at((0, 0)), 1)
        m.set_at((9, 0), 1)
        self.assertEqual(m.get_at((9, 0)), 1)
        self.assertRaises(IndexError, lambda: m.get_at((-1, 0)))
        self.assertRaises(IndexError, lambda: m.set_at((-1, 0), 1))
        self.assertRaises(IndexError, lambda: m.set_at((10, 0), 1))
        self.assertRaises(IndexError, lambda: m.set_at((0, 10), 1))

    def test_fill(self):
        """Ensure a mask can be filled."""
        width, height = (11, 23)
        expected_count = width * height
        expected_size = (width, height)
        mask = pygame.mask.Mask(expected_size)
        mask.fill()
        self.assertEqual(mask.count(), expected_count)
        self.assertEqual(mask.get_size(), expected_size)

    @unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
    def test_fill__bit_boundaries(self):
        """Ensures masks of different sizes are filled correctly.

        Tests masks of different sizes, including:
           -masks 31 to 33 bits wide (32 bit boundaries)
           -masks 63 to 65 bits wide (64 bit boundaries)
        """
        for height in range(1, 4):
            for width in range(1, 66):
                mask = pygame.mask.Mask((width, height))
                expected_count = width * height
                mask.fill()
                self.assertEqual(mask.count(), expected_count, f'size=({width}, {height})')

    def test_clear(self):
        """Ensure a mask can be cleared."""
        expected_count = 0
        expected_size = (13, 27)
        mask = pygame.mask.Mask(expected_size, fill=True)
        mask.clear()
        self.assertEqual(mask.count(), expected_count)
        self.assertEqual(mask.get_size(), expected_size)

    def test_clear__bit_boundaries(self):
        """Ensures masks of different sizes are cleared correctly.

        Tests masks of different sizes, including:
           -masks 31 to 33 bits wide (32 bit boundaries)
           -masks 63 to 65 bits wide (64 bit boundaries)
        """
        expected_count = 0
        for height in range(1, 4):
            for width in range(1, 66):
                mask = pygame.mask.Mask((width, height), fill=True)
                mask.clear()
                self.assertEqual(mask.count(), expected_count, f'size=({width}, {height})')

    @unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
    def test_invert(self):
        """Ensure a mask can be inverted."""
        side = 73
        expected_size = (side, side)
        mask1 = pygame.mask.Mask(expected_size)
        mask2 = pygame.mask.Mask(expected_size, fill=True)
        expected_count1 = side * side
        expected_count2 = 0
        for i in range(side):
            expected_count1 -= 1
            expected_count2 += 1
            pos = (i, i)
            mask1.set_at(pos)
            mask2.set_at(pos, 0)
        mask1.invert()
        mask2.invert()
        self.assertEqual(mask1.count(), expected_count1)
        self.assertEqual(mask2.count(), expected_count2)
        self.assertEqual(mask1.get_size(), expected_size)
        self.assertEqual(mask2.get_size(), expected_size)
        for i in range(side):
            pos = (i, i)
            msg = f'pos={pos}'
            self.assertEqual(mask1.get_at(pos), 0, msg)
            self.assertEqual(mask2.get_at(pos), 1, msg)

    def test_invert__full(self):
        """Ensure a full mask can be inverted."""
        expected_count = 0
        expected_size = (43, 97)
        mask = pygame.mask.Mask(expected_size, fill=True)
        mask.invert()
        self.assertEqual(mask.count(), expected_count)
        self.assertEqual(mask.get_size(), expected_size)

    def test_invert__empty(self):
        """Ensure an empty mask can be inverted."""
        width, height = (43, 97)
        expected_size = (width, height)
        expected_count = width * height
        mask = pygame.mask.Mask(expected_size)
        mask.invert()
        self.assertEqual(mask.count(), expected_count)
        self.assertEqual(mask.get_size(), expected_size)

    @unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
    def test_invert__bit_boundaries(self):
        """Ensures masks of different sizes are inverted correctly.

        Tests masks of different sizes, including:
           -masks 31 to 33 bits wide (32 bit boundaries)
           -masks 63 to 65 bits wide (64 bit boundaries)
        """
        for fill in (True, False):
            for height in range(1, 4):
                for width in range(1, 66):
                    mask = pygame.mask.Mask((width, height), fill=fill)
                    expected_count = 0 if fill else width * height
                    mask.invert()
                    self.assertEqual(mask.count(), expected_count, f'fill={fill}, size=({width}, {height})')

    @unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
    def test_scale(self):
        """Ensure a mask can be scaled."""
        width, height = (43, 61)
        original_size = (width, height)
        for fill in (True, False):
            original_mask = pygame.mask.Mask(original_size, fill=fill)
            original_count = width * height if fill else 0
            for new_w in range(width - 10, width + 10):
                for new_h in range(height - 10, height + 10):
                    expected_size = (new_w, new_h)
                    expected_count = new_w * new_h if fill else 0
                    msg = f'size={expected_size}'
                    mask = original_mask.scale(scale=expected_size)
                    self.assertIsInstance(mask, pygame.mask.Mask, msg)
                    self.assertEqual(mask.count(), expected_count, msg)
                    self.assertEqual(mask.get_size(), expected_size)
                    self.assertEqual(original_mask.count(), original_count, msg)
                    self.assertEqual(original_mask.get_size(), original_size, msg)

    def test_scale__negative_size(self):
        """Ensure scale handles negative sizes correctly."""
        mask = pygame.Mask((100, 100))
        with self.assertRaises(ValueError):
            mask.scale((-1, -1))
        with self.assertRaises(ValueError):
            mask.scale(Vector2(-1, 10))
        with self.assertRaises(ValueError):
            mask.scale((10, -1))

    def test_draw(self):
        """Ensure a mask can be drawn onto another mask.

        Testing the different combinations of full/empty masks:
            (mask1-filled) 1 draw 1 (mask2-filled)
            (mask1-empty)  0 draw 1 (mask2-filled)
            (mask1-filled) 1 draw 0 (mask2-empty)
            (mask1-empty)  0 draw 0 (mask2-empty)
        """
        expected_size = (4, 4)
        offset = (0, 0)
        expected_default = pygame.mask.Mask(expected_size, fill=True)
        expected_masks = {(False, False): pygame.mask.Mask(expected_size)}
        for fill2 in (True, False):
            mask2 = pygame.mask.Mask(expected_size, fill=fill2)
            mask2_count = mask2.count()
            for fill1 in (True, False):
                key = (fill1, fill2)
                msg = f'key={key}'
                mask1 = pygame.mask.Mask(expected_size, fill=fill1)
                expected_mask = expected_masks.get(key, expected_default)
                mask1.draw(mask2, offset)
                assertMaskEqual(self, mask1, expected_mask, msg)
                self.assertEqual(mask2.count(), mask2_count, msg)
                self.assertEqual(mask2.get_size(), expected_size, msg)

    def test_draw__offset(self):
        """Ensure an offset mask can be drawn onto another mask."""
        mask1 = pygame.mask.Mask((65, 3))
        mask2 = pygame.mask.Mask((66, 4), fill=True)
        mask2_count = mask2.count()
        mask2_size = mask2.get_size()
        expected_mask = pygame.Mask(mask1.get_size())
        rect1 = mask1.get_rect()
        rect2 = mask2.get_rect()
        for offset in self.ORIGIN_OFFSETS:
            msg = f'offset={offset}'
            rect2.topleft = offset
            overlap_rect = rect1.clip(rect2)
            expected_mask.clear()
            for x in range(overlap_rect.left, overlap_rect.right):
                for y in range(overlap_rect.top, overlap_rect.bottom):
                    expected_mask.set_at((x, y))
            mask1.clear()
            mask1.draw(other=mask2, offset=offset)
            assertMaskEqual(self, mask1, expected_mask, msg)
            self.assertEqual(mask2.count(), mask2_count, msg)
            self.assertEqual(mask2.get_size(), mask2_size, msg)

    def test_draw__specific_offsets(self):
        """Ensure an offset mask can be drawn onto another mask.

        Testing the specific case of:
            -both masks are wider than 32 bits
            -a positive offset is used
            -the mask calling draw() is wider than the mask passed in
        """
        mask1 = pygame.mask.Mask((65, 5))
        mask2 = pygame.mask.Mask((33, 3), fill=True)
        expected_mask = pygame.Mask(mask1.get_size())
        rect1 = mask1.get_rect()
        rect2 = mask2.get_rect()
        corner_rect = rect1.inflate(-2, -2)
        for corner in ('topleft', 'topright', 'bottomright', 'bottomleft'):
            setattr(rect2, corner, getattr(corner_rect, corner))
            offset = rect2.topleft
            msg = f'offset={offset}'
            overlap_rect = rect1.clip(rect2)
            expected_mask.clear()
            for x in range(overlap_rect.left, overlap_rect.right):
                for y in range(overlap_rect.top, overlap_rect.bottom):
                    expected_mask.set_at((x, y))
            mask1.clear()
            mask1.draw(mask2, offset)
            assertMaskEqual(self, mask1, expected_mask, msg)

    def test_draw__offset_boundary(self):
        """Ensures draw handles offsets and boundaries correctly."""
        mask1 = pygame.mask.Mask((13, 5))
        mask2 = pygame.mask.Mask((7, 3), fill=True)
        mask1_count = mask1.count()
        mask2_count = mask2.count()
        mask1_size = mask1.get_size()
        mask2_size = mask2.get_size()
        offsets = ((mask1_size[0], 0), (0, mask1_size[1]), (-mask2_size[0], 0), (0, -mask2_size[1]))
        for offset in offsets:
            msg = f'offset={offset}'
            mask1.draw(mask2, offset)
            self.assertEqual(mask1.count(), mask1_count, msg)
            self.assertEqual(mask2.count(), mask2_count, msg)
            self.assertEqual(mask1.get_size(), mask1_size, msg)
            self.assertEqual(mask2.get_size(), mask2_size, msg)

    @unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
    def test_draw__bit_boundaries(self):
        """Ensures draw handles masks of different sizes correctly.

        Tests masks of different sizes, including:
           -masks 31 to 33 bits wide (32 bit boundaries)
           -masks 63 to 65 bits wide (64 bit boundaries)
        """
        for height in range(2, 4):
            for width in range(2, 66):
                mask_size = (width, height)
                mask_count = width * height
                mask1 = pygame.mask.Mask(mask_size)
                mask2 = pygame.mask.Mask(mask_size, fill=True)
                expected_mask = pygame.Mask(mask_size)
                rect1 = mask1.get_rect()
                rect2 = mask2.get_rect()
                for offset in self.ORIGIN_OFFSETS:
                    msg = f'size={mask_size}, offset={offset}'
                    rect2.topleft = offset
                    overlap_rect = rect1.clip(rect2)
                    expected_mask.clear()
                    for x in range(overlap_rect.left, overlap_rect.right):
                        for y in range(overlap_rect.top, overlap_rect.bottom):
                            expected_mask.set_at((x, y))
                    mask1.clear()
                    mask1.draw(mask2, offset)
                    assertMaskEqual(self, mask1, expected_mask, msg)
                    self.assertEqual(mask2.count(), mask_count, msg)
                    self.assertEqual(mask2.get_size(), mask_size, msg)

    def test_draw__invalid_mask_arg(self):
        """Ensure draw handles invalid mask arguments correctly."""
        size = (7, 3)
        offset = (0, 0)
        mask = pygame.mask.Mask(size)
        invalid_mask = pygame.Surface(size)
        with self.assertRaises(TypeError):
            mask.draw(invalid_mask, offset)

    def test_draw__invalid_offset_arg(self):
        """Ensure draw handles invalid offset arguments correctly."""
        size = (5, 7)
        offset = '(0, 0)'
        mask1 = pygame.mask.Mask(size)
        mask2 = pygame.mask.Mask(size)
        with self.assertRaises(TypeError):
            mask1.draw(mask2, offset)

    def test_erase(self):
        """Ensure a mask can erase another mask.

        Testing the different combinations of full/empty masks:
            (mask1-filled) 1 erase 1 (mask2-filled)
            (mask1-empty)  0 erase 1 (mask2-filled)
            (mask1-filled) 1 erase 0 (mask2-empty)
            (mask1-empty)  0 erase 0 (mask2-empty)
        """
        expected_size = (4, 4)
        offset = (0, 0)
        expected_default = pygame.mask.Mask(expected_size)
        expected_masks = {(True, False): pygame.mask.Mask(expected_size, fill=True)}
        for fill2 in (True, False):
            mask2 = pygame.mask.Mask(expected_size, fill=fill2)
            mask2_count = mask2.count()
            for fill1 in (True, False):
                key = (fill1, fill2)
                msg = f'key={key}'
                mask1 = pygame.mask.Mask(expected_size, fill=fill1)
                expected_mask = expected_masks.get(key, expected_default)
                mask1.erase(mask2, offset)
                assertMaskEqual(self, mask1, expected_mask, msg)
                self.assertEqual(mask2.count(), mask2_count, msg)
                self.assertEqual(mask2.get_size(), expected_size, msg)

    def test_erase__offset(self):
        """Ensure an offset mask can erase another mask."""
        mask1 = pygame.mask.Mask((65, 3))
        mask2 = pygame.mask.Mask((66, 4), fill=True)
        mask2_count = mask2.count()
        mask2_size = mask2.get_size()
        expected_mask = pygame.Mask(mask1.get_size())
        rect1 = mask1.get_rect()
        rect2 = mask2.get_rect()
        for offset in self.ORIGIN_OFFSETS:
            msg = f'offset={offset}'
            rect2.topleft = offset
            overlap_rect = rect1.clip(rect2)
            expected_mask.fill()
            for x in range(overlap_rect.left, overlap_rect.right):
                for y in range(overlap_rect.top, overlap_rect.bottom):
                    expected_mask.set_at((x, y), 0)
            mask1.fill()
            mask1.erase(other=mask2, offset=offset)
            assertMaskEqual(self, mask1, expected_mask, msg)
            self.assertEqual(mask2.count(), mask2_count, msg)
            self.assertEqual(mask2.get_size(), mask2_size, msg)

    def test_erase__specific_offsets(self):
        """Ensure an offset mask can erase another mask.

        Testing the specific case of:
            -both masks are wider than 32 bits
            -a positive offset is used
            -the mask calling erase() is wider than the mask passed in
        """
        mask1 = pygame.mask.Mask((65, 5))
        mask2 = pygame.mask.Mask((33, 3), fill=True)
        expected_mask = pygame.Mask(mask1.get_size())
        rect1 = mask1.get_rect()
        rect2 = mask2.get_rect()
        corner_rect = rect1.inflate(-2, -2)
        for corner in ('topleft', 'topright', 'bottomright', 'bottomleft'):
            setattr(rect2, corner, getattr(corner_rect, corner))
            offset = rect2.topleft
            msg = f'offset={offset}'
            overlap_rect = rect1.clip(rect2)
            expected_mask.fill()
            for x in range(overlap_rect.left, overlap_rect.right):
                for y in range(overlap_rect.top, overlap_rect.bottom):
                    expected_mask.set_at((x, y), 0)
            mask1.fill()
            mask1.erase(mask2, Vector2(offset))
            assertMaskEqual(self, mask1, expected_mask, msg)

    def test_erase__offset_boundary(self):
        """Ensures erase handles offsets and boundaries correctly."""
        mask1 = pygame.mask.Mask((7, 11), fill=True)
        mask2 = pygame.mask.Mask((3, 13), fill=True)
        mask1_count = mask1.count()
        mask2_count = mask2.count()
        mask1_size = mask1.get_size()
        mask2_size = mask2.get_size()
        offsets = ((mask1_size[0], 0), (0, mask1_size[1]), (-mask2_size[0], 0), (0, -mask2_size[1]))
        for offset in offsets:
            msg = f'offset={offset}'
            mask1.erase(mask2, offset)
            self.assertEqual(mask1.count(), mask1_count, msg)
            self.assertEqual(mask2.count(), mask2_count, msg)
            self.assertEqual(mask1.get_size(), mask1_size, msg)
            self.assertEqual(mask2.get_size(), mask2_size, msg)

    @unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
    def test_erase__bit_boundaries(self):
        """Ensures erase handles masks of different sizes correctly.

        Tests masks of different sizes, including:
           -masks 31 to 33 bits wide (32 bit boundaries)
           -masks 63 to 65 bits wide (64 bit boundaries)
        """
        for height in range(2, 4):
            for width in range(2, 66):
                mask_size = (width, height)
                mask_count = width * height
                mask1 = pygame.mask.Mask(mask_size)
                mask2 = pygame.mask.Mask(mask_size, fill=True)
                expected_mask = pygame.Mask(mask_size)
                rect1 = mask1.get_rect()
                rect2 = mask2.get_rect()
                for offset in self.ORIGIN_OFFSETS:
                    msg = f'size={mask_size}, offset={offset}'
                    rect2.topleft = offset
                    overlap_rect = rect1.clip(rect2)
                    expected_mask.fill()
                    for x in range(overlap_rect.left, overlap_rect.right):
                        for y in range(overlap_rect.top, overlap_rect.bottom):
                            expected_mask.set_at((x, y), 0)
                    mask1.fill()
                    mask1.erase(mask2, offset)
                    assertMaskEqual(self, mask1, expected_mask, msg)
                    self.assertEqual(mask2.count(), mask_count, msg)
                    self.assertEqual(mask2.get_size(), mask_size, msg)

    def test_erase__invalid_mask_arg(self):
        """Ensure erase handles invalid mask arguments correctly."""
        size = (3, 7)
        offset = (0, 0)
        mask = pygame.mask.Mask(size)
        invalid_mask = pygame.Surface(size)
        with self.assertRaises(TypeError):
            mask.erase(invalid_mask, offset)

    def test_erase__invalid_offset_arg(self):
        """Ensure erase handles invalid offset arguments correctly."""
        size = (7, 5)
        offset = '(0, 0)'
        mask1 = pygame.mask.Mask(size)
        mask2 = pygame.mask.Mask(size)
        with self.assertRaises(TypeError):
            mask1.erase(mask2, offset)

    def test_count(self):
        """Ensure a mask's set bits are correctly counted."""
        side = 67
        expected_size = (side, side)
        expected_count = 0
        mask = pygame.mask.Mask(expected_size)
        for i in range(side):
            expected_count += 1
            mask.set_at((i, i))
        count = mask.count()
        self.assertEqual(count, expected_count)
        self.assertEqual(mask.get_size(), expected_size)

    @unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
    def test_count__bit_boundaries(self):
        """Ensures the set bits of different sized masks are counted correctly.

        Tests masks of different sizes, including:
           -masks 31 to 33 bits wide (32 bit boundaries)
           -masks 63 to 65 bits wide (64 bit boundaries)
        """
        for fill in (True, False):
            for height in range(1, 4):
                for width in range(1, 66):
                    mask = pygame.mask.Mask((width, height), fill=fill)
                    expected_count = width * height if fill else 0
                    for pos in ((x, y) for y in range(height) for x in range(width)):
                        if fill:
                            mask.set_at(pos, 0)
                            expected_count -= 1
                        else:
                            mask.set_at(pos, 1)
                            expected_count += 1
                        count = mask.count()
                        self.assertEqual(count, expected_count, f'fill={fill}, size=({width}, {height}), pos={pos}')

    def test_count__full_mask(self):
        """Ensure a full mask's set bits are correctly counted."""
        width, height = (17, 97)
        expected_size = (width, height)
        expected_count = width * height
        mask = pygame.mask.Mask(expected_size, fill=True)
        count = mask.count()
        self.assertEqual(count, expected_count)
        self.assertEqual(mask.get_size(), expected_size)

    def test_count__empty_mask(self):
        """Ensure an empty mask's set bits are correctly counted."""
        expected_count = 0
        expected_size = (13, 27)
        mask = pygame.mask.Mask(expected_size)
        count = mask.count()
        self.assertEqual(count, expected_count)
        self.assertEqual(mask.get_size(), expected_size)

    def test_centroid(self):
        """Ensure a filled mask's centroid is correctly calculated."""
        mask = pygame.mask.Mask((5, 7), fill=True)
        expected_centroid = mask.get_rect().center
        centroid = mask.centroid()
        self.assertEqual(centroid, expected_centroid)

    def test_centroid__empty_mask(self):
        """Ensure an empty mask's centroid is correctly calculated."""
        expected_centroid = (0, 0)
        expected_size = (101, 103)
        mask = pygame.mask.Mask(expected_size)
        centroid = mask.centroid()
        self.assertEqual(centroid, expected_centroid)
        self.assertEqual(mask.get_size(), expected_size)

    def test_centroid__single_row(self):
        """Ensure a mask's centroid is correctly calculated
        when setting points along a single row."""
        width, height = (5, 7)
        mask = pygame.mask.Mask((width, height))
        for y in range(height):
            mask.clear()
            for x in range(width):
                mask.set_at((x, y))
                expected_centroid = (x // 2, y)
                centroid = mask.centroid()
                self.assertEqual(centroid, expected_centroid)

    def test_centroid__two_rows(self):
        """Ensure a mask's centroid is correctly calculated
        when setting points along two rows."""
        width, height = (5, 7)
        mask = pygame.mask.Mask((width, height))
        for y in range(1, height):
            mask.clear()
            for x in range(width):
                mask.set_at((x, 0))
                mask.set_at((x, y))
                expected_centroid = (x // 2, y // 2)
                centroid = mask.centroid()
                self.assertEqual(centroid, expected_centroid)

    def test_centroid__single_column(self):
        """Ensure a mask's centroid is correctly calculated
        when setting points along a single column."""
        width, height = (5, 7)
        mask = pygame.mask.Mask((width, height))
        for x in range(width):
            mask.clear()
            for y in range(height):
                mask.set_at((x, y))
                expected_centroid = (x, y // 2)
                centroid = mask.centroid()
                self.assertEqual(centroid, expected_centroid)

    def test_centroid__two_columns(self):
        """Ensure a mask's centroid is correctly calculated
        when setting points along two columns."""
        width, height = (5, 7)
        mask = pygame.mask.Mask((width, height))
        for x in range(1, width):
            mask.clear()
            for y in range(height):
                mask.set_at((0, y))
                mask.set_at((x, y))
                expected_centroid = (x // 2, y // 2)
                centroid = mask.centroid()
                self.assertEqual(centroid, expected_centroid)

    def test_centroid__all_corners(self):
        """Ensure a mask's centroid is correctly calculated
        when its corners are set."""
        mask = pygame.mask.Mask((5, 7))
        expected_centroid = mask.get_rect().center
        for corner in corners(mask):
            mask.set_at(corner)
        centroid = mask.centroid()
        self.assertEqual(centroid, expected_centroid)

    def test_centroid__two_corners(self):
        """Ensure a mask's centroid is correctly calculated
        when only two corners are set."""
        mask = pygame.mask.Mask((5, 7))
        mask_rect = mask.get_rect()
        mask_corners = corners(mask)
        for i, corner1 in enumerate(mask_corners):
            for corner2 in mask_corners[i + 1:]:
                mask.clear()
                mask.set_at(corner1)
                mask.set_at(corner2)
                if corner1[0] == corner2[0]:
                    expected_centroid = (corner1[0], abs(corner1[1] - corner2[1]) // 2)
                elif corner1[1] == corner2[1]:
                    expected_centroid = (abs(corner1[0] - corner2[0]) // 2, corner1[1])
                else:
                    expected_centroid = mask_rect.center
                centroid = mask.centroid()
                self.assertEqual(centroid, expected_centroid)

    def test_angle(self):
        """Ensure a mask's orientation angle is correctly calculated."""
        expected_angle = -45.0
        expected_size = (100, 100)
        surface = pygame.Surface(expected_size)
        mask = pygame.mask.from_surface(surface)
        angle = mask.angle()
        self.assertIsInstance(angle, float)
        self.assertEqual(angle, expected_angle)

    def test_angle__empty_mask(self):
        """Ensure an empty mask's angle is correctly calculated."""
        expected_angle = 0.0
        expected_size = (107, 43)
        mask = pygame.mask.Mask(expected_size)
        angle = mask.angle()
        self.assertIsInstance(angle, float)
        self.assertAlmostEqual(angle, expected_angle)
        self.assertEqual(mask.get_size(), expected_size)

    def test_drawing(self):
        """Test fill, clear, invert, draw, erase"""
        m = pygame.Mask((100, 100))
        self.assertEqual(m.count(), 0)
        m.fill()
        self.assertEqual(m.count(), 10000)
        m2 = pygame.Mask((10, 10), fill=True)
        m.erase(m2, (50, 50))
        self.assertEqual(m.count(), 9900)
        m.invert()
        self.assertEqual(m.count(), 100)
        m.draw(m2, (0, 0))
        self.assertEqual(m.count(), 200)
        m.clear()
        self.assertEqual(m.count(), 0)

    def test_outline(self):
        """ """
        m = pygame.Mask((20, 20))
        self.assertEqual(m.outline(), [])
        m.set_at((10, 10), 1)
        self.assertEqual(m.outline(), [(10, 10)])
        m.set_at((10, 12), 1)
        self.assertEqual(m.outline(10), [(10, 10)])
        m.set_at((11, 11), 1)
        self.assertEqual(m.outline(), [(10, 10), (11, 11), (10, 12), (11, 11), (10, 10)])
        self.assertEqual(m.outline(every=2), [(10, 10), (10, 12), (10, 10)])

    def test_convolve__size(self):
        sizes = [(1, 1), (31, 31), (32, 32), (100, 100)]
        for s1 in sizes:
            m1 = pygame.Mask(s1)
            for s2 in sizes:
                m2 = pygame.Mask(s2)
                o = m1.convolve(m2)
                self.assertIsInstance(o, pygame.mask.Mask)
                for i in (0, 1):
                    self.assertEqual(o.get_size()[i], m1.get_size()[i] + m2.get_size()[i] - 1)

    def test_convolve__point_identities(self):
        """Convolving with a single point is the identity, while convolving a point with something flips it."""
        m = random_mask((100, 100))
        k = pygame.Mask((1, 1))
        k.set_at((0, 0))
        convolve_mask = m.convolve(k)
        self.assertIsInstance(convolve_mask, pygame.mask.Mask)
        assertMaskEqual(self, m, convolve_mask)
        convolve_mask = k.convolve(k.convolve(m))
        self.assertIsInstance(convolve_mask, pygame.mask.Mask)
        assertMaskEqual(self, m, convolve_mask)

    def test_convolve__with_output(self):
        """checks that convolution modifies only the correct portion of the output"""
        m = random_mask((10, 10))
        k = pygame.Mask((2, 2))
        k.set_at((0, 0))
        o = pygame.Mask((50, 50))
        test = pygame.Mask((50, 50))
        m.convolve(k, o)
        test.draw(m, (1, 1))
        self.assertIsInstance(o, pygame.mask.Mask)
        assertMaskEqual(self, o, test)
        o.clear()
        test.clear()
        m.convolve(other=k, output=o, offset=Vector2(10, 10))
        test.draw(m, (11, 11))
        self.assertIsInstance(o, pygame.mask.Mask)
        assertMaskEqual(self, o, test)

    def test_convolve__out_of_range(self):
        full = pygame.Mask((2, 2), fill=True)
        pts_data = (((0, 3), 0), ((0, 2), 3), ((-2, -2), 1), ((-3, -3), 0))
        for pt, expected_count in pts_data:
            convolve_mask = full.convolve(full, None, pt)
            self.assertIsInstance(convolve_mask, pygame.mask.Mask)
            self.assertEqual(convolve_mask.count(), expected_count)

    @unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
    def test_convolve(self):
        """Tests the definition of convolution"""
        m1 = random_mask((100, 100))
        m2 = random_mask((100, 100))
        conv = m1.convolve(m2)
        self.assertIsInstance(conv, pygame.mask.Mask)
        for i in range(conv.get_size()[0]):
            for j in range(conv.get_size()[1]):
                self.assertEqual(conv.get_at((i, j)) == 0, m1.overlap(m2, (i - 99, j - 99)) is None)

    def _draw_component_pattern_box(self, mask, size, pos, inverse=False):
        pattern = pygame.mask.Mask((size, size), fill=True)
        pattern.set_at((size // 2, size // 2), 0)
        if inverse:
            mask.erase(pattern, pos)
            pattern.invert()
        else:
            mask.draw(pattern, pos)
        return pattern

    def _draw_component_pattern_x(self, mask, size, pos, inverse=False):
        pattern = pygame.mask.Mask((size, size))
        ymax = size - 1
        for y in range(size):
            for x in range(size):
                if x in [y, ymax - y]:
                    pattern.set_at((x, y))
        if inverse:
            mask.erase(pattern, pos)
            pattern.invert()
        else:
            mask.draw(pattern, pos)
        return pattern

    def _draw_component_pattern_plus(self, mask, size, pos, inverse=False):
        pattern = pygame.mask.Mask((size, size))
        xmid = ymid = size // 2
        for y in range(size):
            for x in range(size):
                if x == xmid or y == ymid:
                    pattern.set_at((x, y))
        if inverse:
            mask.erase(pattern, pos)
            pattern.invert()
        else:
            mask.draw(pattern, pos)
        return pattern

    def test_connected_component(self):
        """Ensure a mask's connected component is correctly calculated."""
        width, height = (41, 27)
        expected_size = (width, height)
        original_mask = pygame.mask.Mask(expected_size)
        patterns = []
        offset = (0, 0)
        pattern = self._draw_component_pattern_x(original_mask, 3, offset)
        patterns.append((pattern, offset))
        size = 4
        offset = (width - size, 0)
        pattern = self._draw_component_pattern_plus(original_mask, size, offset)
        patterns.append((pattern, offset))
        offset = (width // 2, height // 2)
        pattern = self._draw_component_pattern_box(original_mask, 7, offset)
        patterns.append((pattern, offset))
        expected_pattern, expected_offset = patterns[-1]
        expected_count = expected_pattern.count()
        original_count = sum((p.count() for p, _ in patterns))
        mask = original_mask.connected_component()
        self.assertIsInstance(mask, pygame.mask.Mask)
        self.assertEqual(mask.count(), expected_count)
        self.assertEqual(mask.get_size(), expected_size)
        self.assertEqual(mask.overlap_area(expected_pattern, expected_offset), expected_count)
        self.assertEqual(original_mask.count(), original_count)
        self.assertEqual(original_mask.get_size(), expected_size)
        for pattern, offset in patterns:
            self.assertEqual(original_mask.overlap_area(pattern, offset), pattern.count())

    def test_connected_component__full_mask(self):
        """Ensure a mask's connected component is correctly calculated
        when the mask is full.
        """
        expected_size = (23, 31)
        original_mask = pygame.mask.Mask(expected_size, fill=True)
        expected_count = original_mask.count()
        mask = original_mask.connected_component()
        self.assertIsInstance(mask, pygame.mask.Mask)
        self.assertEqual(mask.count(), expected_count)
        self.assertEqual(mask.get_size(), expected_size)
        self.assertEqual(original_mask.count(), expected_count)
        self.assertEqual(original_mask.get_size(), expected_size)

    def test_connected_component__empty_mask(self):
        """Ensure a mask's connected component is correctly calculated
        when the mask is empty.
        """
        expected_size = (37, 43)
        original_mask = pygame.mask.Mask(expected_size)
        original_count = original_mask.count()
        expected_count = 0
        mask = original_mask.connected_component()
        self.assertIsInstance(mask, pygame.mask.Mask)
        self.assertEqual(mask.count(), expected_count)
        self.assertEqual(mask.get_size(), expected_size)
        self.assertEqual(original_mask.count(), original_count)
        self.assertEqual(original_mask.get_size(), expected_size)

    def test_connected_component__one_set_bit(self):
        """Ensure a mask's connected component is correctly calculated
        when the coordinate's bit is set with a connected component of 1 bit.
        """
        width, height = (71, 67)
        expected_size = (width, height)
        original_mask = pygame.mask.Mask(expected_size, fill=True)
        xset, yset = (width // 2, height // 2)
        set_pos = (xset, yset)
        expected_offset = (xset - 1, yset - 1)
        expected_pattern = self._draw_component_pattern_box(original_mask, 3, expected_offset, inverse=True)
        expected_count = 1
        original_count = original_mask.count()
        mask = original_mask.connected_component(set_pos)
        self.assertIsInstance(mask, pygame.mask.Mask)
        self.assertEqual(mask.count(), expected_count)
        self.assertEqual(mask.get_size(), expected_size)
        self.assertEqual(mask.overlap_area(expected_pattern, expected_offset), expected_count)
        self.assertEqual(original_mask.count(), original_count)
        self.assertEqual(original_mask.get_size(), expected_size)
        self.assertEqual(original_mask.overlap_area(expected_pattern, expected_offset), expected_count)

    def test_connected_component__multi_set_bits(self):
        """Ensure a mask's connected component is correctly calculated
        when the coordinate's bit is set with a connected component of > 1 bit.
        """
        expected_size = (113, 67)
        original_mask = pygame.mask.Mask(expected_size)
        p_width, p_height = (11, 13)
        set_pos = xset, yset = (11, 21)
        expected_offset = (xset - 1, yset - 1)
        expected_pattern = pygame.mask.Mask((p_width, p_height), fill=True)
        for y in range(3, p_height):
            for x in range(1, p_width):
                if x in [y, y - 3, p_width - 4]:
                    expected_pattern.set_at((x, y), 0)
        expected_count = expected_pattern.count()
        original_mask.draw(expected_pattern, expected_offset)
        mask = original_mask.connected_component(set_pos)
        self.assertIsInstance(mask, pygame.mask.Mask)
        self.assertEqual(mask.count(), expected_count)
        self.assertEqual(mask.get_size(), expected_size)
        self.assertEqual(mask.overlap_area(expected_pattern, expected_offset), expected_count)
        self.assertEqual(original_mask.count(), expected_count)
        self.assertEqual(original_mask.get_size(), expected_size)
        self.assertEqual(original_mask.overlap_area(expected_pattern, expected_offset), expected_count)

    def test_connected_component__unset_bit(self):
        """Ensure a mask's connected component is correctly calculated
        when the coordinate's bit is unset.
        """
        width, height = (109, 101)
        expected_size = (width, height)
        original_mask = pygame.mask.Mask(expected_size, fill=True)
        unset_pos = (width // 2, height // 2)
        original_mask.set_at(unset_pos, 0)
        original_count = original_mask.count()
        expected_count = 0
        mask = original_mask.connected_component(unset_pos)
        self.assertIsInstance(mask, pygame.mask.Mask)
        self.assertEqual(mask.count(), expected_count)
        self.assertEqual(mask.get_size(), expected_size)
        self.assertEqual(original_mask.count(), original_count)
        self.assertEqual(original_mask.get_size(), expected_size)
        self.assertEqual(original_mask.get_at(unset_pos), 0)

    def test_connected_component__out_of_bounds(self):
        """Ensure connected_component() checks bounds."""
        width, height = (19, 11)
        original_size = (width, height)
        original_mask = pygame.mask.Mask(original_size, fill=True)
        original_count = original_mask.count()
        for pos in ((0, -1), (-1, 0), (0, height + 1), (width + 1, 0)):
            with self.assertRaises(IndexError):
                mask = original_mask.connected_component(pos)
            self.assertEqual(original_mask.count(), original_count)
            self.assertEqual(original_mask.get_size(), original_size)

    def test_connected_components(self):
        """ """
        m = pygame.Mask((10, 10))
        self.assertListEqual(m.connected_components(), [])
        comp = m.connected_component()
        self.assertEqual(m.count(), comp.count())
        m.set_at((0, 0), 1)
        m.set_at((1, 1), 1)
        comp = m.connected_component()
        comps = m.connected_components()
        comps1 = m.connected_components(1)
        comps2 = m.connected_components(2)
        comps3 = m.connected_components(3)
        self.assertEqual(comp.count(), comps[0].count())
        self.assertEqual(comps1[0].count(), 2)
        self.assertEqual(comps2[0].count(), 2)
        self.assertListEqual(comps3, [])
        m.set_at((9, 9), 1)
        comp = m.connected_component()
        comp1 = m.connected_component((1, 1))
        comp2 = m.connected_component((2, 2))
        comps = m.connected_components()
        comps1 = m.connected_components(1)
        comps2 = m.connected_components(minimum=2)
        comps3 = m.connected_components(3)
        self.assertEqual(comp.count(), 2)
        self.assertEqual(comp1.count(), 2)
        self.assertEqual(comp2.count(), 0)
        self.assertEqual(len(comps), 2)
        self.assertEqual(len(comps1), 2)
        self.assertEqual(len(comps2), 1)
        self.assertEqual(len(comps3), 0)
        for mask in comps:
            self.assertIsInstance(mask, pygame.mask.Mask)

    def test_connected_components__negative_min_with_empty_mask(self):
        """Ensures connected_components() properly handles negative min values
        when the mask is empty.

        Negative and zero values for the min parameter (minimum number of bits
        per connected component) equate to setting it to one.
        """
        expected_comps = []
        mask_count = 0
        mask_size = (65, 13)
        mask = pygame.mask.Mask(mask_size)
        connected_comps = mask.connected_components(-1)
        self.assertListEqual(connected_comps, expected_comps)
        self.assertEqual(mask.count(), mask_count)
        self.assertEqual(mask.get_size(), mask_size)

    def test_connected_components__negative_min_with_full_mask(self):
        """Ensures connected_components() properly handles negative min values
        when the mask is full.

        Negative and zero values for the min parameter (minimum number of bits
        per connected component) equate to setting it to one.
        """
        mask_size = (64, 11)
        mask = pygame.mask.Mask(mask_size, fill=True)
        mask_count = mask.count()
        expected_len = 1
        connected_comps = mask.connected_components(-2)
        self.assertEqual(len(connected_comps), expected_len)
        assertMaskEqual(self, connected_comps[0], mask)
        self.assertEqual(mask.count(), mask_count)
        self.assertEqual(mask.get_size(), mask_size)

    def test_connected_components__negative_min_with_some_bits_set(self):
        """Ensures connected_components() properly handles negative min values
        when the mask has some bits set.

        Negative and zero values for the min parameter (minimum number of bits
        per connected component) equate to setting it to one.
        """
        mask_size = (64, 12)
        mask = pygame.mask.Mask(mask_size)
        expected_comps = {}
        for corner in corners(mask):
            mask.set_at(corner)
            new_mask = pygame.mask.Mask(mask_size)
            new_mask.set_at(corner)
            expected_comps[corner] = new_mask
        center = (mask_size[0] // 2, mask_size[1] // 2)
        mask.set_at(center)
        new_mask = pygame.mask.Mask(mask_size)
        new_mask.set_at(center)
        expected_comps[center] = new_mask
        mask_count = mask.count()
        connected_comps = mask.connected_components(-3)
        self.assertEqual(len(connected_comps), len(expected_comps))
        for comp in connected_comps:
            found = False
            for pt in tuple(expected_comps.keys()):
                if comp.get_at(pt):
                    found = True
                    assertMaskEqual(self, comp, expected_comps[pt])
                    del expected_comps[pt]
                    break
            self.assertTrue(found, f'missing component for pt={pt}')
        self.assertEqual(mask.count(), mask_count)
        self.assertEqual(mask.get_size(), mask_size)

    @unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
    def test_get_bounding_rects(self):
        """Ensures get_bounding_rects works correctly."""
        mask_data = []
        mask_data.append(((10, 10), (((0, 0), (1, 0), (0, 1)), ((0, 3),), ((3, 3),))))
        mask_data.append(((4, 2), (((0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (3, 1)),)))
        mask_data.append(((5, 3), (((2, 0), (1, 1), (2, 1), (3, 1), (2, 2)),)))
        mask_data.append(((5, 3), (((3, 0), (2, 1), (1, 2)),)))
        mask_data.append(((5, 2), (((3, 0), (4, 0), (0, 1), (1, 1), (2, 1), (3, 1)),)))
        mask_data.append(((5, 3), (((0, 0),), ((4, 0),), ((2, 1),), ((0, 2),), ((4, 2),))))
        for size, rect_point_tuples in mask_data:
            rects = []
            mask = pygame.Mask(size)
            for rect_points in rect_point_tuples:
                rects.append(create_bounding_rect(rect_points))
                for pt in rect_points:
                    mask.set_at(pt)
            expected_rects = sorted(rects, key=tuple)
            rects = mask.get_bounding_rects()
            self.assertListEqual(sorted(mask.get_bounding_rects(), key=tuple), expected_rects, f'size={size}')

    @unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
    def test_to_surface(self):
        """Ensures empty and full masks can be drawn onto surfaces."""
        expected_ref_count = 3
        size = (33, 65)
        surface = pygame.Surface(size, SRCALPHA, 32)
        surface_color = pygame.Color('red')
        test_fills = ((pygame.Color('white'), True), (pygame.Color('black'), False))
        for expected_color, fill in test_fills:
            surface.fill(surface_color)
            mask = pygame.mask.Mask(size, fill=fill)
            to_surface = mask.to_surface(surface)
            self.assertIs(to_surface, surface)
            if not IS_PYPY:
                self.assertEqual(sys.getrefcount(to_surface), expected_ref_count)
            self.assertEqual(to_surface.get_size(), size)
            assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__create_surface(self):
        """Ensures empty and full masks can be drawn onto a created surface."""
        expected_ref_count = 2
        expected_flag = SRCALPHA
        expected_depth = 32
        size = (33, 65)
        test_fills = ((pygame.Color('white'), True), (pygame.Color('black'), False))
        for expected_color, fill in test_fills:
            mask = pygame.mask.Mask(size, fill=fill)
            for use_arg in (True, False):
                if use_arg:
                    to_surface = mask.to_surface(None)
                else:
                    to_surface = mask.to_surface()
                self.assertIsInstance(to_surface, pygame.Surface)
                if not IS_PYPY:
                    self.assertEqual(sys.getrefcount(to_surface), expected_ref_count)
                self.assertTrue(to_surface.get_flags() & expected_flag)
                self.assertEqual(to_surface.get_bitsize(), expected_depth)
                self.assertEqual(to_surface.get_size(), size)
                assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__surface_param(self):
        """Ensures to_surface accepts a surface arg/kwarg."""
        expected_ref_count = 4
        expected_color = pygame.Color('white')
        surface_color = pygame.Color('red')
        size = (5, 3)
        mask = pygame.mask.Mask(size, fill=True)
        surface = pygame.Surface(size)
        kwargs = {'surface': surface}
        for use_kwargs in (True, False):
            surface.fill(surface_color)
            if use_kwargs:
                to_surface = mask.to_surface(**kwargs)
            else:
                to_surface = mask.to_surface(kwargs['surface'])
            self.assertIs(to_surface, surface)
            if not IS_PYPY:
                self.assertEqual(sys.getrefcount(to_surface), expected_ref_count)
            self.assertEqual(to_surface.get_size(), size)
            assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__setsurface_param(self):
        """Ensures to_surface accepts a setsurface arg/kwarg."""
        expected_ref_count = 2
        expected_flag = SRCALPHA
        expected_depth = 32
        expected_color = pygame.Color('red')
        size = (5, 3)
        mask = pygame.mask.Mask(size, fill=True)
        setsurface = pygame.Surface(size, expected_flag, expected_depth)
        setsurface.fill(expected_color)
        kwargs = {'setsurface': setsurface}
        for use_kwargs in (True, False):
            if use_kwargs:
                to_surface = mask.to_surface(**kwargs)
            else:
                to_surface = mask.to_surface(None, kwargs['setsurface'])
            self.assertIsInstance(to_surface, pygame.Surface)
            if not IS_PYPY:
                self.assertEqual(sys.getrefcount(to_surface), expected_ref_count)
            self.assertTrue(to_surface.get_flags() & expected_flag)
            self.assertEqual(to_surface.get_bitsize(), expected_depth)
            self.assertEqual(to_surface.get_size(), size)
            assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__unsetsurface_param(self):
        """Ensures to_surface accepts a unsetsurface arg/kwarg."""
        expected_ref_count = 2
        expected_flag = SRCALPHA
        expected_depth = 32
        expected_color = pygame.Color('red')
        size = (5, 3)
        mask = pygame.mask.Mask(size)
        unsetsurface = pygame.Surface(size, expected_flag, expected_depth)
        unsetsurface.fill(expected_color)
        kwargs = {'unsetsurface': unsetsurface}
        for use_kwargs in (True, False):
            if use_kwargs:
                to_surface = mask.to_surface(**kwargs)
            else:
                to_surface = mask.to_surface(None, None, kwargs['unsetsurface'])
            self.assertIsInstance(to_surface, pygame.Surface)
            if not IS_PYPY:
                self.assertEqual(sys.getrefcount(to_surface), expected_ref_count)
            self.assertTrue(to_surface.get_flags() & expected_flag)
            self.assertEqual(to_surface.get_bitsize(), expected_depth)
            self.assertEqual(to_surface.get_size(), size)
            assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__setcolor_param(self):
        """Ensures to_surface accepts a setcolor arg/kwarg."""
        expected_ref_count = 2
        expected_flag = SRCALPHA
        expected_depth = 32
        expected_color = pygame.Color('red')
        size = (5, 3)
        mask = pygame.mask.Mask(size, fill=True)
        kwargs = {'setcolor': expected_color}
        for use_kwargs in (True, False):
            if use_kwargs:
                to_surface = mask.to_surface(**kwargs)
            else:
                to_surface = mask.to_surface(None, None, None, kwargs['setcolor'])
            self.assertIsInstance(to_surface, pygame.Surface)
            if not IS_PYPY:
                self.assertEqual(sys.getrefcount(to_surface), expected_ref_count)
            self.assertTrue(to_surface.get_flags() & expected_flag)
            self.assertEqual(to_surface.get_bitsize(), expected_depth)
            self.assertEqual(to_surface.get_size(), size)
            assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__setcolor_default(self):
        """Ensures the default setcolor is correct."""
        expected_color = pygame.Color('white')
        size = (3, 7)
        mask = pygame.mask.Mask(size, fill=True)
        to_surface = mask.to_surface(surface=None, setsurface=None, unsetsurface=None, unsetcolor=None)
        self.assertEqual(to_surface.get_size(), size)
        assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__unsetcolor_param(self):
        """Ensures to_surface accepts a unsetcolor arg/kwarg."""
        expected_ref_count = 2
        expected_flag = SRCALPHA
        expected_depth = 32
        expected_color = pygame.Color('red')
        size = (5, 3)
        mask = pygame.mask.Mask(size)
        kwargs = {'unsetcolor': expected_color}
        for use_kwargs in (True, False):
            if use_kwargs:
                to_surface = mask.to_surface(**kwargs)
            else:
                to_surface = mask.to_surface(None, None, None, None, kwargs['unsetcolor'])
            self.assertIsInstance(to_surface, pygame.Surface)
            if not IS_PYPY:
                self.assertEqual(sys.getrefcount(to_surface), expected_ref_count)
            self.assertTrue(to_surface.get_flags() & expected_flag)
            self.assertEqual(to_surface.get_bitsize(), expected_depth)
            self.assertEqual(to_surface.get_size(), size)
            assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__unsetcolor_default(self):
        """Ensures the default unsetcolor is correct."""
        expected_color = pygame.Color('black')
        size = (3, 7)
        mask = pygame.mask.Mask(size)
        to_surface = mask.to_surface(surface=None, setsurface=None, unsetsurface=None, setcolor=None)
        self.assertEqual(to_surface.get_size(), size)
        assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__dest_param(self):
        """Ensures to_surface accepts a dest arg/kwarg."""
        expected_ref_count = 2
        expected_flag = SRCALPHA
        expected_depth = 32
        default_surface_color = (0, 0, 0, 0)
        default_unsetcolor = pygame.Color('black')
        dest = (0, 0)
        size = (5, 3)
        mask = pygame.mask.Mask(size)
        kwargs = {'dest': dest}
        for use_kwargs in (True, False):
            if use_kwargs:
                expected_color = default_unsetcolor
                to_surface = mask.to_surface(**kwargs)
            else:
                expected_color = default_surface_color
                to_surface = mask.to_surface(None, None, None, None, None, kwargs['dest'])
            self.assertIsInstance(to_surface, pygame.Surface)
            if not IS_PYPY:
                self.assertEqual(sys.getrefcount(to_surface), expected_ref_count)
            self.assertTrue(to_surface.get_flags() & expected_flag)
            self.assertEqual(to_surface.get_bitsize(), expected_depth)
            self.assertEqual(to_surface.get_size(), size)
            assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__dest_default(self):
        """Ensures the default dest is correct."""
        expected_color = pygame.Color('white')
        surface_color = pygame.Color('red')
        mask_size = (3, 2)
        mask = pygame.mask.Mask(mask_size, fill=True)
        mask_rect = mask.get_rect()
        surf_size = (mask_size[0] + 2, mask_size[1] + 1)
        surface = pygame.Surface(surf_size, SRCALPHA, 32)
        surface.fill(surface_color)
        to_surface = mask.to_surface(surface, setsurface=None, unsetsurface=None, unsetcolor=None)
        self.assertIs(to_surface, surface)
        self.assertEqual(to_surface.get_size(), surf_size)
        assertSurfaceFilled(self, to_surface, expected_color, mask_rect)
        assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, mask_rect)

    @unittest.expectedFailure
    def test_to_surface__area_param(self):
        """Ensures to_surface accepts an area arg/kwarg."""
        expected_ref_count = 2
        expected_flag = SRCALPHA
        expected_depth = 32
        default_surface_color = (0, 0, 0, 0)
        default_unsetcolor = pygame.Color('black')
        size = (5, 3)
        mask = pygame.mask.Mask(size)
        kwargs = {'area': mask.get_rect()}
        for use_kwargs in (True, False):
            if use_kwargs:
                expected_color = default_unsetcolor
                to_surface = mask.to_surface(**kwargs)
            else:
                expected_color = default_surface_color
                to_surface = mask.to_surface(None, None, None, None, None, (0, 0), kwargs['area'])
            self.assertIsInstance(to_surface, pygame.Surface)
            if not IS_PYPY:
                self.assertEqual(sys.getrefcount(to_surface), expected_ref_count)
            self.assertTrue(to_surface.get_flags() & expected_flag)
            self.assertEqual(to_surface.get_bitsize(), expected_depth)
            self.assertEqual(to_surface.get_size(), size)
            assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__area_default(self):
        """Ensures the default area is correct."""
        expected_color = pygame.Color('white')
        surface_color = pygame.Color('red')
        mask_size = (3, 2)
        mask = pygame.mask.Mask(mask_size, fill=True)
        mask_rect = mask.get_rect()
        surf_size = (mask_size[0] + 2, mask_size[1] + 1)
        surface = pygame.Surface(surf_size, SRCALPHA, 32)
        surface.fill(surface_color)
        to_surface = mask.to_surface(surface, setsurface=None, unsetsurface=None, unsetcolor=None)
        self.assertIs(to_surface, surface)
        self.assertEqual(to_surface.get_size(), surf_size)
        assertSurfaceFilled(self, to_surface, expected_color, mask_rect)
        assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, mask_rect)

    def test_to_surface__kwargs(self):
        """Ensures to_surface accepts the correct kwargs."""
        expected_color = pygame.Color('white')
        size = (5, 3)
        mask = pygame.mask.Mask(size, fill=True)
        surface = pygame.Surface(size)
        surface_color = pygame.Color('red')
        setsurface = surface.copy()
        setsurface.fill(expected_color)
        test_data = ((None, None), ('dest', (0, 0)), ('unsetcolor', pygame.Color('yellow')), ('setcolor', expected_color), ('unsetsurface', surface.copy()), ('setsurface', setsurface), ('surface', surface))
        kwargs = dict(test_data)
        for name, _ in test_data:
            kwargs.pop(name)
            surface.fill(surface_color)
            to_surface = mask.to_surface(**kwargs)
            self.assertEqual(to_surface.get_size(), size)
            assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__kwargs_create_surface(self):
        """Ensures to_surface accepts the correct kwargs
        when creating a surface.
        """
        expected_color = pygame.Color('black')
        size = (5, 3)
        mask = pygame.mask.Mask(size)
        setsurface = pygame.Surface(size, SRCALPHA, 32)
        setsurface_color = pygame.Color('red')
        setsurface.fill(setsurface_color)
        unsetsurface = setsurface.copy()
        unsetsurface.fill(expected_color)
        test_data = ((None, None), ('dest', (0, 0)), ('unsetcolor', expected_color), ('setcolor', pygame.Color('yellow')), ('unsetsurface', unsetsurface), ('setsurface', setsurface), ('surface', None))
        kwargs = dict(test_data)
        for name, _ in test_data:
            kwargs.pop(name)
            to_surface = mask.to_surface(**kwargs)
            self.assertIsInstance(to_surface, pygame.Surface)
            self.assertEqual(to_surface.get_size(), size)
            assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__kwargs_order_independent(self):
        """Ensures to_surface kwargs are not order dependent."""
        expected_color = pygame.Color('blue')
        size = (3, 2)
        mask = pygame.mask.Mask(size, fill=True)
        surface = pygame.Surface(size)
        to_surface = mask.to_surface(dest=(0, 0), setcolor=expected_color, unsetcolor=None, surface=surface, unsetsurface=pygame.Surface(size), setsurface=None)
        self.assertIs(to_surface, surface)
        self.assertEqual(to_surface.get_size(), size)
        assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__args_invalid_types(self):
        """Ensures to_surface detects invalid kwarg types."""
        size = (3, 2)
        mask = pygame.mask.Mask(size, fill=True)
        invalid_surf = pygame.Color('green')
        invalid_color = pygame.Surface(size)
        with self.assertRaises(TypeError):
            mask.to_surface(None, None, None, None, None, (0,))
        with self.assertRaises(TypeError):
            mask.to_surface(None, None, None, None, invalid_color)
        with self.assertRaises(TypeError):
            mask.to_surface(None, None, None, invalid_color, None)
        with self.assertRaises(TypeError):
            mask.to_surface(None, None, invalid_surf, None, None)
        with self.assertRaises(TypeError):
            mask.to_surface(None, invalid_surf, None, None, None)
        with self.assertRaises(TypeError):
            mask.to_surface(invalid_surf, None, None, None, None)

    def test_to_surface__kwargs_invalid_types(self):
        """Ensures to_surface detects invalid kwarg types."""
        size = (3, 2)
        mask = pygame.mask.Mask(size)
        valid_kwargs = {'surface': pygame.Surface(size), 'setsurface': pygame.Surface(size), 'unsetsurface': pygame.Surface(size), 'setcolor': pygame.Color('green'), 'unsetcolor': pygame.Color('green'), 'dest': (0, 0)}
        invalid_kwargs = {'surface': (1, 2, 3, 4), 'setsurface': pygame.Color('green'), 'unsetsurface': ((1, 2), (2, 1)), 'setcolor': pygame.Mask((1, 2)), 'unsetcolor': pygame.Surface((2, 2)), 'dest': (0, 0, 0)}
        kwarg_order = ('surface', 'setsurface', 'unsetsurface', 'setcolor', 'unsetcolor', 'dest')
        for kwarg in kwarg_order:
            kwargs = dict(valid_kwargs)
            kwargs[kwarg] = invalid_kwargs[kwarg]
            with self.assertRaises(TypeError):
                mask.to_surface(**kwargs)

    def test_to_surface__kwargs_invalid_name(self):
        """Ensures to_surface detects invalid kwarg names."""
        mask = pygame.mask.Mask((3, 2))
        kwargs = {'setcolour': pygame.Color('red')}
        with self.assertRaises(TypeError):
            mask.to_surface(**kwargs)

    def test_to_surface__args_and_kwargs(self):
        """Ensures to_surface accepts a combination of args/kwargs"""
        size = (5, 3)
        surface_color = pygame.Color('red')
        setsurface_color = pygame.Color('yellow')
        unsetsurface_color = pygame.Color('blue')
        setcolor = pygame.Color('green')
        unsetcolor = pygame.Color('cyan')
        surface = pygame.Surface(size, SRCALPHA, 32)
        setsurface = surface.copy()
        unsetsurface = surface.copy()
        setsurface.fill(setsurface_color)
        unsetsurface.fill(unsetsurface_color)
        mask = pygame.mask.Mask(size, fill=True)
        expected_color = setsurface_color
        test_data = ((None, None), ('surface', surface), ('setsurface', setsurface), ('unsetsurface', unsetsurface), ('setcolor', setcolor), ('unsetcolor', unsetcolor), ('dest', (0, 0)))
        args = []
        kwargs = dict(test_data)
        for name, value in test_data:
            if name is not None:
                args.append(value)
            kwargs.pop(name)
            surface.fill(surface_color)
            to_surface = mask.to_surface(*args, **kwargs)
            assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__valid_setcolor_formats(self):
        """Ensures to_surface handles valid setcolor formats correctly."""
        size = (5, 3)
        mask = pygame.mask.Mask(size, fill=True)
        surface = pygame.Surface(size, SRCALPHA, 32)
        expected_color = pygame.Color('green')
        test_colors = ((0, 255, 0), (0, 255, 0, 255), surface.map_rgb(expected_color), expected_color, 'green', '#00FF00FF', '0x00FF00FF')
        for setcolor in test_colors:
            to_surface = mask.to_surface(setcolor=setcolor)
            assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__valid_unsetcolor_formats(self):
        """Ensures to_surface handles valid unsetcolor formats correctly."""
        size = (5, 3)
        mask = pygame.mask.Mask(size)
        surface = pygame.Surface(size, SRCALPHA, 32)
        expected_color = pygame.Color('green')
        test_colors = ((0, 255, 0), (0, 255, 0, 255), surface.map_rgb(expected_color), expected_color, 'green', '#00FF00FF', '0x00FF00FF')
        for unsetcolor in test_colors:
            to_surface = mask.to_surface(unsetcolor=unsetcolor)
            assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__invalid_setcolor_formats(self):
        """Ensures to_surface handles invalid setcolor formats correctly."""
        mask = pygame.mask.Mask((5, 3))
        for setcolor in ('green color', '#00FF00FF0', '0x00FF00FF0', (1, 2)):
            with self.assertRaises(ValueError):
                mask.to_surface(setcolor=setcolor)
        for setcolor in (pygame.Surface((1, 2)), pygame.Mask((2, 1)), 1.1):
            with self.assertRaises(TypeError):
                mask.to_surface(setcolor=setcolor)

    def test_to_surface__invalid_unsetcolor_formats(self):
        """Ensures to_surface handles invalid unsetcolor formats correctly."""
        mask = pygame.mask.Mask((5, 3))
        for unsetcolor in ('green color', '#00FF00FF0', '0x00FF00FF0', (1, 2)):
            with self.assertRaises(ValueError):
                mask.to_surface(unsetcolor=unsetcolor)
        for unsetcolor in (pygame.Surface((1, 2)), pygame.Mask((2, 1)), 1.1):
            with self.assertRaises(TypeError):
                mask.to_surface(unsetcolor=unsetcolor)

    def test_to_surface__valid_dest_formats(self):
        """Ensures to_surface handles valid dest formats correctly."""
        expected_color = pygame.Color('white')
        mask = pygame.mask.Mask((3, 5), fill=True)
        dests = ((0, 0), [0, 0], Vector2(0, 0), (0, 0, 100, 100), pygame.Rect((0, 0), (10, 10)))
        for dest in dests:
            to_surface = mask.to_surface(dest=dest)
            assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__invalid_dest_formats(self):
        """Ensures to_surface handles invalid dest formats correctly."""
        mask = pygame.mask.Mask((3, 5))
        invalid_dests = ((0,), (0, 0, 0), {0, 1}, {0: 1}, Rect)
        for dest in invalid_dests:
            with self.assertRaises(TypeError):
                mask.to_surface(dest=dest)

    def test_to_surface__negative_sized_dest_rect(self):
        """Ensures to_surface correctly handles negative sized dest rects."""
        expected_color = pygame.Color('white')
        mask = pygame.mask.Mask((3, 5), fill=True)
        dests = (pygame.Rect((0, 0), (10, -10)), pygame.Rect((0, 0), (-10, 10)), pygame.Rect((0, 0), (-10, -10)))
        for dest in dests:
            to_surface = mask.to_surface(dest=dest)
            assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__zero_sized_dest_rect(self):
        """Ensures to_surface correctly handles zero sized dest rects."""
        expected_color = pygame.Color('white')
        mask = pygame.mask.Mask((3, 5), fill=True)
        dests = (pygame.Rect((0, 0), (0, 10)), pygame.Rect((0, 0), (10, 0)), pygame.Rect((0, 0), (0, 0)))
        for dest in dests:
            to_surface = mask.to_surface(dest=dest)
            assertSurfaceFilled(self, to_surface, expected_color)

    @unittest.expectedFailure
    def test_to_surface__valid_area_formats(self):
        """Ensures to_surface handles valid area formats correctly."""
        size = (3, 5)
        surface_color = pygame.Color('red')
        expected_color = pygame.Color('white')
        surface = pygame.Surface(size)
        mask = pygame.mask.Mask(size, fill=True)
        area_pos = (0, 0)
        area_size = (2, 1)
        areas = ((area_pos[0], area_pos[1], area_size[0], area_size[1]), (area_pos, area_size), (area_pos, list(area_size)), (list(area_pos), area_size), (list(area_pos), list(area_size)), [area_pos[0], area_pos[1], area_size[0], area_size[1]], [area_pos, area_size], [area_pos, list(area_size)], [list(area_pos), area_size], [list(area_pos), list(area_size)], pygame.Rect(area_pos, area_size))
        for area in areas:
            surface.fill(surface_color)
            area_rect = pygame.Rect(area)
            to_surface = mask.to_surface(surface, area=area)
            assertSurfaceFilled(self, to_surface, expected_color, area_rect)
            assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, area_rect)

    @unittest.expectedFailure
    def test_to_surface__invalid_area_formats(self):
        """Ensures to_surface handles invalid area formats correctly."""
        mask = pygame.mask.Mask((3, 5))
        invalid_areas = ((0,), (0, 0), (0, 0, 1), ((0, 0), (1,)), ((0,), (1, 1)), {0, 1, 2, 3}, {0: 1, 2: 3}, Rect)
        for area in invalid_areas:
            with self.assertRaisesRegex(TypeError, 'invalid area argument'):
                unused_to_surface = mask.to_surface(area=area)

    @unittest.expectedFailure
    def test_to_surface__negative_sized_area_rect(self):
        """Ensures to_surface correctly handles negative sized area rects."""
        size = (3, 5)
        surface_color = pygame.Color('red')
        expected_color = pygame.Color('white')
        surface = pygame.Surface(size)
        mask = pygame.mask.Mask(size)
        mask.set_at((0, 0))
        areas = (pygame.Rect((0, 1), (1, -1)), pygame.Rect((1, 0), (-1, 1)), pygame.Rect((1, 1), (-1, -1)))
        for area in areas:
            surface.fill(surface_color)
            to_surface = mask.to_surface(surface, area=area)
            assertSurfaceFilled(self, to_surface, expected_color, area)
            assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, area)

    @unittest.expectedFailure
    def test_to_surface__zero_sized_area_rect(self):
        """Ensures to_surface correctly handles zero sized area rects."""
        size = (3, 5)
        expected_color = pygame.Color('red')
        surface = pygame.Surface(size)
        mask = pygame.mask.Mask(size, fill=True)
        areas = (pygame.Rect((0, 0), (0, 1)), pygame.Rect((0, 0), (1, 0)), pygame.Rect((0, 0), (0, 0)))
        for area in areas:
            surface.fill(expected_color)
            to_surface = mask.to_surface(surface, area=area)
            assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__default_surface_with_param_combinations(self):
        """Ensures to_surface works with a default surface value
        and combinations of other parameters.

        This tests many different parameter combinations with full and empty
        masks.
        """
        expected_ref_count = 2
        expected_flag = SRCALPHA
        expected_depth = 32
        size = (5, 3)
        dest = (0, 0)
        default_surface_color = (0, 0, 0, 0)
        setsurface_color = pygame.Color('yellow')
        unsetsurface_color = pygame.Color('blue')
        setcolor = pygame.Color('green')
        unsetcolor = pygame.Color('cyan')
        setsurface = pygame.Surface(size, expected_flag, expected_depth)
        unsetsurface = setsurface.copy()
        setsurface.fill(setsurface_color)
        unsetsurface.fill(unsetsurface_color)
        kwargs = {'setsurface': None, 'unsetsurface': None, 'setcolor': None, 'unsetcolor': None, 'dest': None}
        for fill in (True, False):
            mask = pygame.mask.Mask(size, fill=fill)
            for setsurface_param in (setsurface, None):
                kwargs['setsurface'] = setsurface_param
                for unsetsurface_param in (unsetsurface, None):
                    kwargs['unsetsurface'] = unsetsurface_param
                    for setcolor_param in (setcolor, None):
                        kwargs['setcolor'] = setcolor_param
                        for unsetcolor_param in (unsetcolor, None):
                            kwargs['unsetcolor'] = unsetcolor_param
                            for dest_param in (dest, None):
                                if dest_param is None:
                                    kwargs.pop('dest', None)
                                else:
                                    kwargs['dest'] = dest_param
                                if fill:
                                    if setsurface_param is not None:
                                        expected_color = setsurface_color
                                    elif setcolor_param is not None:
                                        expected_color = setcolor
                                    else:
                                        expected_color = default_surface_color
                                elif unsetsurface_param is not None:
                                    expected_color = unsetsurface_color
                                elif unsetcolor_param is not None:
                                    expected_color = unsetcolor
                                else:
                                    expected_color = default_surface_color
                                to_surface = mask.to_surface(**kwargs)
                                self.assertIsInstance(to_surface, pygame.Surface)
                                if not IS_PYPY:
                                    self.assertEqual(sys.getrefcount(to_surface), expected_ref_count)
                                self.assertTrue(to_surface.get_flags() & expected_flag)
                                self.assertEqual(to_surface.get_bitsize(), expected_depth)
                                self.assertEqual(to_surface.get_size(), size)
                                assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__surface_with_param_combinations(self):
        """Ensures to_surface works with a surface value
        and combinations of other parameters.

        This tests many different parameter combinations with full and empty
        masks.
        """
        expected_ref_count = 4
        expected_flag = SRCALPHA
        expected_depth = 32
        size = (5, 3)
        dest = (0, 0)
        surface_color = pygame.Color('red')
        setsurface_color = pygame.Color('yellow')
        unsetsurface_color = pygame.Color('blue')
        setcolor = pygame.Color('green')
        unsetcolor = pygame.Color('cyan')
        surface = pygame.Surface(size, expected_flag, expected_depth)
        setsurface = surface.copy()
        unsetsurface = surface.copy()
        setsurface.fill(setsurface_color)
        unsetsurface.fill(unsetsurface_color)
        kwargs = {'surface': surface, 'setsurface': None, 'unsetsurface': None, 'setcolor': None, 'unsetcolor': None, 'dest': None}
        for fill in (True, False):
            mask = pygame.mask.Mask(size, fill=fill)
            for setsurface_param in (setsurface, None):
                kwargs['setsurface'] = setsurface_param
                for unsetsurface_param in (unsetsurface, None):
                    kwargs['unsetsurface'] = unsetsurface_param
                    for setcolor_param in (setcolor, None):
                        kwargs['setcolor'] = setcolor_param
                        for unsetcolor_param in (unsetcolor, None):
                            kwargs['unsetcolor'] = unsetcolor_param
                            surface.fill(surface_color)
                            for dest_param in (dest, None):
                                if dest_param is None:
                                    kwargs.pop('dest', None)
                                else:
                                    kwargs['dest'] = dest_param
                                if fill:
                                    if setsurface_param is not None:
                                        expected_color = setsurface_color
                                    elif setcolor_param is not None:
                                        expected_color = setcolor
                                    else:
                                        expected_color = surface_color
                                elif unsetsurface_param is not None:
                                    expected_color = unsetsurface_color
                                elif unsetcolor_param is not None:
                                    expected_color = unsetcolor
                                else:
                                    expected_color = surface_color
                                to_surface = mask.to_surface(**kwargs)
                                self.assertIs(to_surface, surface)
                                if not IS_PYPY:
                                    self.assertEqual(sys.getrefcount(to_surface), expected_ref_count)
                                self.assertTrue(to_surface.get_flags() & expected_flag)
                                self.assertEqual(to_surface.get_bitsize(), expected_depth)
                                self.assertEqual(to_surface.get_size(), size)
                                assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__set_and_unset_bits(self):
        """Ensures that to_surface works correctly with with set/unset bits
        when using the defaults for setcolor and unsetcolor.
        """
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        width, height = size = (10, 20)
        mask = pygame.mask.Mask(size)
        mask_rect = mask.get_rect()
        surface = pygame.Surface(size)
        surface_color = pygame.Color('red')
        for pos in ((x, y) for x in range(width) for y in range(x & 1, height, 2)):
            mask.set_at(pos)
        for dest in self.ORIGIN_OFFSETS:
            mask_rect.topleft = dest
            surface.fill(surface_color)
            to_surface = mask.to_surface(surface, dest=dest)
            to_surface.lock()
            for pos in ((x, y) for x in range(width) for y in range(height)):
                mask_pos = (pos[0] - dest[0], pos[1] - dest[1])
                if not mask_rect.collidepoint(pos):
                    expected_color = surface_color
                elif mask.get_at(mask_pos):
                    expected_color = default_setcolor
                else:
                    expected_color = default_unsetcolor
                self.assertEqual(to_surface.get_at(pos), expected_color, (dest, pos))
            to_surface.unlock()

    def test_to_surface__set_and_unset_bits_with_setsurface_unsetsurface(self):
        """Ensures that to_surface works correctly with with set/unset bits
        when using setsurface and unsetsurface.
        """
        width, height = size = (10, 20)
        mask = pygame.mask.Mask(size)
        mask_rect = mask.get_rect()
        surface = pygame.Surface(size)
        surface_color = pygame.Color('red')
        setsurface = surface.copy()
        setsurface_color = pygame.Color('green')
        setsurface.fill(setsurface_color)
        unsetsurface = surface.copy()
        unsetsurface_color = pygame.Color('blue')
        unsetsurface.fill(unsetsurface_color)
        for pos in ((x, y) for x in range(width) for y in range(x & 1, height, 2)):
            mask.set_at(pos)
        for dest in self.ORIGIN_OFFSETS:
            mask_rect.topleft = dest
            for disable_color_params in (True, False):
                surface.fill(surface_color)
                if disable_color_params:
                    to_surface = mask.to_surface(surface, dest=dest, setsurface=setsurface, unsetsurface=unsetsurface, setcolor=None, unsetcolor=None)
                else:
                    to_surface = mask.to_surface(surface, dest=dest, setsurface=setsurface, unsetsurface=unsetsurface)
                to_surface.lock()
                for pos in ((x, y) for x in range(width) for y in range(height)):
                    mask_pos = (pos[0] - dest[0], pos[1] - dest[1])
                    if not mask_rect.collidepoint(pos):
                        expected_color = surface_color
                    elif mask.get_at(mask_pos):
                        expected_color = setsurface_color
                    else:
                        expected_color = unsetsurface_color
                    self.assertEqual(to_surface.get_at(pos), expected_color)
                to_surface.unlock()

    def test_to_surface__surface_narrower_than_mask(self):
        """Ensures that surfaces narrower than the mask work correctly.

        For this test the surface's width is less than the mask's width.
        """
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        mask_size = (10, 20)
        narrow_size = (6, 20)
        surface = pygame.Surface(narrow_size)
        surface_color = pygame.Color('red')
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            surface.fill(surface_color)
            expected_color = default_setcolor if fill else default_unsetcolor
            to_surface = mask.to_surface(surface)
            self.assertIs(to_surface, surface)
            self.assertEqual(to_surface.get_size(), narrow_size)
            assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__setsurface_narrower_than_mask(self):
        """Ensures that setsurfaces narrower than the mask work correctly.

        For this test the setsurface's width is less than the mask's width.
        """
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        mask_size = (10, 20)
        narrow_size = (6, 20)
        setsurface = pygame.Surface(narrow_size, SRCALPHA, 32)
        setsurface_color = pygame.Color('red')
        setsurface.fill(setsurface_color)
        setsurface_rect = setsurface.get_rect()
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            to_surface = mask.to_surface(setsurface=setsurface)
            self.assertIsInstance(to_surface, pygame.Surface)
            self.assertEqual(to_surface.get_size(), mask_size)
            if fill:
                assertSurfaceFilled(self, to_surface, setsurface_color, setsurface_rect)
                assertSurfaceFilledIgnoreArea(self, to_surface, default_setcolor, setsurface_rect)
            else:
                assertSurfaceFilled(self, to_surface, default_unsetcolor)

    def test_to_surface__unsetsurface_narrower_than_mask(self):
        """Ensures that unsetsurfaces narrower than the mask work correctly.

        For this test the unsetsurface's width is less than the mask's width.
        """
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        mask_size = (10, 20)
        narrow_size = (6, 20)
        unsetsurface = pygame.Surface(narrow_size, SRCALPHA, 32)
        unsetsurface_color = pygame.Color('red')
        unsetsurface.fill(unsetsurface_color)
        unsetsurface_rect = unsetsurface.get_rect()
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            to_surface = mask.to_surface(unsetsurface=unsetsurface)
            self.assertIsInstance(to_surface, pygame.Surface)
            self.assertEqual(to_surface.get_size(), mask_size)
            if fill:
                assertSurfaceFilled(self, to_surface, default_setcolor)
            else:
                assertSurfaceFilled(self, to_surface, unsetsurface_color, unsetsurface_rect)
                assertSurfaceFilledIgnoreArea(self, to_surface, default_unsetcolor, unsetsurface_rect)

    def test_to_surface__setsurface_narrower_than_mask_and_colors_none(self):
        """Ensures that setsurfaces narrower than the mask work correctly
        when setcolor and unsetcolor are set to None.

        For this test the setsurface's width is less than the mask's width.
        """
        default_surface_color = (0, 0, 0, 0)
        mask_size = (10, 20)
        narrow_size = (6, 20)
        setsurface = pygame.Surface(narrow_size, SRCALPHA, 32)
        setsurface_color = pygame.Color('red')
        setsurface.fill(setsurface_color)
        setsurface_rect = setsurface.get_rect()
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            to_surface = mask.to_surface(setsurface=setsurface, setcolor=None, unsetcolor=None)
            self.assertIsInstance(to_surface, pygame.Surface)
            self.assertEqual(to_surface.get_size(), mask_size)
            if fill:
                assertSurfaceFilled(self, to_surface, setsurface_color, setsurface_rect)
                assertSurfaceFilledIgnoreArea(self, to_surface, default_surface_color, setsurface_rect)
            else:
                assertSurfaceFilled(self, to_surface, default_surface_color)

    def test_to_surface__unsetsurface_narrower_than_mask_and_colors_none(self):
        """Ensures that unsetsurfaces narrower than the mask work correctly
        when setcolor and unsetcolor are set to None.

        For this test the unsetsurface's width is less than the mask's width.
        """
        default_surface_color = (0, 0, 0, 0)
        mask_size = (10, 20)
        narrow_size = (6, 20)
        unsetsurface = pygame.Surface(narrow_size, SRCALPHA, 32)
        unsetsurface_color = pygame.Color('red')
        unsetsurface.fill(unsetsurface_color)
        unsetsurface_rect = unsetsurface.get_rect()
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            to_surface = mask.to_surface(unsetsurface=unsetsurface, setcolor=None, unsetcolor=None)
            self.assertIsInstance(to_surface, pygame.Surface)
            self.assertEqual(to_surface.get_size(), mask_size)
            if fill:
                assertSurfaceFilled(self, to_surface, default_surface_color)
            else:
                assertSurfaceFilled(self, to_surface, unsetsurface_color, unsetsurface_rect)
                assertSurfaceFilledIgnoreArea(self, to_surface, default_surface_color, unsetsurface_rect)

    def test_to_surface__surface_wider_than_mask(self):
        """Ensures that surfaces wider than the mask work correctly.

        For this test the surface's width is greater than the mask's width.
        """
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        mask_size = (6, 15)
        wide_size = (11, 15)
        surface = pygame.Surface(wide_size)
        surface_color = pygame.Color('red')
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            mask_rect = mask.get_rect()
            surface.fill(surface_color)
            expected_color = default_setcolor if fill else default_unsetcolor
            to_surface = mask.to_surface(surface)
            self.assertIs(to_surface, surface)
            self.assertEqual(to_surface.get_size(), wide_size)
            assertSurfaceFilled(self, to_surface, expected_color, mask_rect)
            assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, mask_rect)

    def test_to_surface__setsurface_wider_than_mask(self):
        """Ensures that setsurfaces wider than the mask work correctly.

        For this test the setsurface's width is greater than the mask's width.
        """
        default_unsetcolor = pygame.Color('black')
        mask_size = (6, 15)
        wide_size = (11, 15)
        setsurface = pygame.Surface(wide_size, SRCALPHA, 32)
        setsurface_color = pygame.Color('red')
        setsurface.fill(setsurface_color)
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            expected_color = setsurface_color if fill else default_unsetcolor
            to_surface = mask.to_surface(setsurface=setsurface)
            self.assertIsInstance(to_surface, pygame.Surface)
            self.assertEqual(to_surface.get_size(), mask_size)
            assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__unsetsurface_wider_than_mask(self):
        """Ensures that unsetsurfaces wider than the mask work correctly.

        For this test the unsetsurface's width is greater than the mask's
        width.
        """
        default_setcolor = pygame.Color('white')
        mask_size = (6, 15)
        wide_size = (11, 15)
        unsetsurface = pygame.Surface(wide_size, SRCALPHA, 32)
        unsetsurface_color = pygame.Color('red')
        unsetsurface.fill(unsetsurface_color)
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            expected_color = default_setcolor if fill else unsetsurface_color
            to_surface = mask.to_surface(unsetsurface=unsetsurface)
            self.assertIsInstance(to_surface, pygame.Surface)
            self.assertEqual(to_surface.get_size(), mask_size)
            assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__surface_shorter_than_mask(self):
        """Ensures that surfaces shorter than the mask work correctly.

        For this test the surface's height is less than the mask's height.
        """
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        mask_size = (10, 11)
        short_size = (10, 6)
        surface = pygame.Surface(short_size)
        surface_color = pygame.Color('red')
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            surface.fill(surface_color)
            expected_color = default_setcolor if fill else default_unsetcolor
            to_surface = mask.to_surface(surface)
            self.assertIs(to_surface, surface)
            self.assertEqual(to_surface.get_size(), short_size)
            assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__setsurface_shorter_than_mask(self):
        """Ensures that setsurfaces shorter than the mask work correctly.

        For this test the setsurface's height is less than the mask's height.
        """
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        mask_size = (10, 11)
        short_size = (10, 6)
        setsurface = pygame.Surface(short_size, SRCALPHA, 32)
        setsurface_color = pygame.Color('red')
        setsurface.fill(setsurface_color)
        setsurface_rect = setsurface.get_rect()
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            to_surface = mask.to_surface(setsurface=setsurface)
            self.assertIsInstance(to_surface, pygame.Surface)
            self.assertEqual(to_surface.get_size(), mask_size)
            if fill:
                assertSurfaceFilled(self, to_surface, setsurface_color, setsurface_rect)
                assertSurfaceFilledIgnoreArea(self, to_surface, default_setcolor, setsurface_rect)
            else:
                assertSurfaceFilled(self, to_surface, default_unsetcolor)

    def test_to_surface__unsetsurface_shorter_than_mask(self):
        """Ensures that unsetsurfaces shorter than the mask work correctly.

        For this test the unsetsurface's height is less than the mask's height.
        """
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        mask_size = (10, 11)
        short_size = (10, 6)
        unsetsurface = pygame.Surface(short_size, SRCALPHA, 32)
        unsetsurface_color = pygame.Color('red')
        unsetsurface.fill(unsetsurface_color)
        unsetsurface_rect = unsetsurface.get_rect()
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            to_surface = mask.to_surface(unsetsurface=unsetsurface)
            self.assertIsInstance(to_surface, pygame.Surface)
            self.assertEqual(to_surface.get_size(), mask_size)
            if fill:
                assertSurfaceFilled(self, to_surface, default_setcolor)
            else:
                assertSurfaceFilled(self, to_surface, unsetsurface_color, unsetsurface_rect)
                assertSurfaceFilledIgnoreArea(self, to_surface, default_unsetcolor, unsetsurface_rect)

    def test_to_surface__setsurface_shorter_than_mask_and_colors_none(self):
        """Ensures that setsurfaces shorter than the mask work correctly
        when setcolor and unsetcolor are set to None.

        For this test the setsurface's height is less than the mask's height.
        """
        default_surface_color = (0, 0, 0, 0)
        mask_size = (10, 11)
        short_size = (10, 6)
        setsurface = pygame.Surface(short_size, SRCALPHA, 32)
        setsurface_color = pygame.Color('red')
        setsurface.fill(setsurface_color)
        setsurface_rect = setsurface.get_rect()
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            to_surface = mask.to_surface(setsurface=setsurface, setcolor=None, unsetcolor=None)
            self.assertIsInstance(to_surface, pygame.Surface)
            self.assertEqual(to_surface.get_size(), mask_size)
            if fill:
                assertSurfaceFilled(self, to_surface, setsurface_color, setsurface_rect)
                assertSurfaceFilledIgnoreArea(self, to_surface, default_surface_color, setsurface_rect)
            else:
                assertSurfaceFilled(self, to_surface, default_surface_color)

    def test_to_surface__unsetsurface_shorter_than_mask_and_colors_none(self):
        """Ensures that unsetsurfaces shorter than the mask work correctly
        when setcolor and unsetcolor are set to None.

        For this test the unsetsurface's height is less than the mask's height.
        """
        default_surface_color = (0, 0, 0, 0)
        mask_size = (10, 11)
        short_size = (10, 6)
        unsetsurface = pygame.Surface(short_size, SRCALPHA, 32)
        unsetsurface_color = pygame.Color('red')
        unsetsurface.fill(unsetsurface_color)
        unsetsurface_rect = unsetsurface.get_rect()
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            to_surface = mask.to_surface(unsetsurface=unsetsurface, setcolor=None, unsetcolor=None)
            self.assertIsInstance(to_surface, pygame.Surface)
            self.assertEqual(to_surface.get_size(), mask_size)
            if fill:
                assertSurfaceFilled(self, to_surface, default_surface_color)
            else:
                assertSurfaceFilled(self, to_surface, unsetsurface_color, unsetsurface_rect)
                assertSurfaceFilledIgnoreArea(self, to_surface, default_surface_color, unsetsurface_rect)

    def test_to_surface__surface_taller_than_mask(self):
        """Ensures that surfaces taller than the mask work correctly.

        For this test the surface's height is greater than the mask's height.
        """
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        mask_size = (10, 6)
        tall_size = (10, 11)
        surface = pygame.Surface(tall_size)
        surface_color = pygame.Color('red')
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            mask_rect = mask.get_rect()
            surface.fill(surface_color)
            expected_color = default_setcolor if fill else default_unsetcolor
            to_surface = mask.to_surface(surface)
            self.assertIs(to_surface, surface)
            self.assertEqual(to_surface.get_size(), tall_size)
            assertSurfaceFilled(self, to_surface, expected_color, mask_rect)
            assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, mask_rect)

    def test_to_surface__setsurface_taller_than_mask(self):
        """Ensures that setsurfaces taller than the mask work correctly.

        For this test the setsurface's height is greater than the mask's
        height.
        """
        default_unsetcolor = pygame.Color('black')
        mask_size = (10, 6)
        tall_size = (10, 11)
        setsurface = pygame.Surface(tall_size, SRCALPHA, 32)
        setsurface_color = pygame.Color('red')
        setsurface.fill(setsurface_color)
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            expected_color = setsurface_color if fill else default_unsetcolor
            to_surface = mask.to_surface(setsurface=setsurface)
            self.assertIsInstance(to_surface, pygame.Surface)
            self.assertEqual(to_surface.get_size(), mask_size)
            assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__unsetsurface_taller_than_mask(self):
        """Ensures that unsetsurfaces taller than the mask work correctly.

        For this test the unsetsurface's height is greater than the mask's
        height.
        """
        default_setcolor = pygame.Color('white')
        mask_size = (10, 6)
        tall_size = (10, 11)
        unsetsurface = pygame.Surface(tall_size, SRCALPHA, 32)
        unsetsurface_color = pygame.Color('red')
        unsetsurface.fill(unsetsurface_color)
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            expected_color = default_setcolor if fill else unsetsurface_color
            to_surface = mask.to_surface(unsetsurface=unsetsurface)
            self.assertIsInstance(to_surface, pygame.Surface)
            self.assertEqual(to_surface.get_size(), mask_size)
            assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__surface_wider_and_taller_than_mask(self):
        """Ensures that surfaces wider and taller than the mask work correctly.

        For this test the surface's width is greater than the mask's width and
        the surface's height is greater than the mask's height.
        """
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        mask_size = (6, 8)
        wide_tall_size = (11, 15)
        surface = pygame.Surface(wide_tall_size)
        surface_color = pygame.Color('red')
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            mask_rect = mask.get_rect()
            surface.fill(surface_color)
            expected_color = default_setcolor if fill else default_unsetcolor
            to_surface = mask.to_surface(surface)
            self.assertIs(to_surface, surface)
            self.assertEqual(to_surface.get_size(), wide_tall_size)
            assertSurfaceFilled(self, to_surface, expected_color, mask_rect)
            assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, mask_rect)

    def test_to_surface__setsurface_wider_and_taller_than_mask(self):
        """Ensures that setsurfaces wider and taller than the mask work
        correctly.

        For this test the setsurface's width is greater than the mask's width
        and the setsurface's height is greater than the mask's height.
        """
        default_unsetcolor = pygame.Color('black')
        mask_size = (6, 8)
        wide_tall_size = (11, 15)
        setsurface = pygame.Surface(wide_tall_size, SRCALPHA, 32)
        setsurface_color = pygame.Color('red')
        setsurface.fill(setsurface_color)
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            expected_color = setsurface_color if fill else default_unsetcolor
            to_surface = mask.to_surface(setsurface=setsurface)
            self.assertIsInstance(to_surface, pygame.Surface)
            self.assertEqual(to_surface.get_size(), mask_size)
            assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__unsetsurface_wider_and_taller_than_mask(self):
        """Ensures that unsetsurfaces wider and taller than the mask work
        correctly.

        For this test the unsetsurface's width is greater than the mask's width
        and the unsetsurface's height is greater than the mask's height.
        """
        default_setcolor = pygame.Color('white')
        mask_size = (6, 8)
        wide_tall_size = (11, 15)
        unsetsurface = pygame.Surface(wide_tall_size, SRCALPHA, 32)
        unsetsurface_color = pygame.Color('red')
        unsetsurface.fill(unsetsurface_color)
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            expected_color = default_setcolor if fill else unsetsurface_color
            to_surface = mask.to_surface(unsetsurface=unsetsurface)
            self.assertIsInstance(to_surface, pygame.Surface)
            self.assertEqual(to_surface.get_size(), mask_size)
            assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__surface_wider_and_shorter_than_mask(self):
        """Ensures that surfaces wider and shorter than the mask work
        correctly.

        For this test the surface's width is greater than the mask's width and
        the surface's height is less than the mask's height.
        """
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        mask_size = (7, 11)
        wide_short_size = (13, 6)
        surface = pygame.Surface(wide_short_size)
        surface_color = pygame.Color('red')
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            mask_rect = mask.get_rect()
            surface.fill(surface_color)
            expected_color = default_setcolor if fill else default_unsetcolor
            to_surface = mask.to_surface(surface)
            self.assertIs(to_surface, surface)
            self.assertEqual(to_surface.get_size(), wide_short_size)
            assertSurfaceFilled(self, to_surface, expected_color, mask_rect)
            assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, mask_rect)

    def test_to_surface__setsurface_wider_and_shorter_than_mask(self):
        """Ensures that setsurfaces wider and shorter than the mask work
        correctly.

        For this test the setsurface's width is greater than the mask's width
        and the setsurface's height is less than the mask's height.
        """
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        mask_size = (7, 11)
        wide_short_size = (10, 6)
        setsurface = pygame.Surface(wide_short_size, SRCALPHA, 32)
        setsurface_color = pygame.Color('red')
        setsurface.fill(setsurface_color)
        setsurface_rect = setsurface.get_rect()
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            to_surface = mask.to_surface(setsurface=setsurface)
            self.assertIsInstance(to_surface, pygame.Surface)
            self.assertEqual(to_surface.get_size(), mask_size)
            if fill:
                assertSurfaceFilled(self, to_surface, setsurface_color, setsurface_rect)
                assertSurfaceFilledIgnoreArea(self, to_surface, default_setcolor, setsurface_rect)
            else:
                assertSurfaceFilled(self, to_surface, default_unsetcolor)

    def test_to_surface__unsetsurface_wider_and_shorter_than_mask(self):
        """Ensures that unsetsurfaces wider and shorter than the mask work
        correctly.

        For this test the unsetsurface's width is greater than the mask's width
        and the unsetsurface's height is less than the mask's height.
        """
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        mask_size = (7, 11)
        wide_short_size = (10, 6)
        unsetsurface = pygame.Surface(wide_short_size, SRCALPHA, 32)
        unsetsurface_color = pygame.Color('red')
        unsetsurface.fill(unsetsurface_color)
        unsetsurface_rect = unsetsurface.get_rect()
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            to_surface = mask.to_surface(unsetsurface=unsetsurface)
            self.assertIsInstance(to_surface, pygame.Surface)
            self.assertEqual(to_surface.get_size(), mask_size)
            if fill:
                assertSurfaceFilled(self, to_surface, default_setcolor)
            else:
                assertSurfaceFilled(self, to_surface, unsetsurface_color, unsetsurface_rect)
                assertSurfaceFilledIgnoreArea(self, to_surface, default_unsetcolor, unsetsurface_rect)

    def test_to_surface__surface_narrower_and_taller_than_mask(self):
        """Ensures that surfaces narrower and taller than the mask work
        correctly.

        For this test the surface's width is less than the mask's width and
        the surface's height is greater than the mask's height.
        """
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        mask_size = (10, 8)
        narrow_tall_size = (6, 15)
        surface = pygame.Surface(narrow_tall_size)
        surface_color = pygame.Color('red')
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            mask_rect = mask.get_rect()
            surface.fill(surface_color)
            expected_color = default_setcolor if fill else default_unsetcolor
            to_surface = mask.to_surface(surface)
            self.assertIs(to_surface, surface)
            self.assertEqual(to_surface.get_size(), narrow_tall_size)
            assertSurfaceFilled(self, to_surface, expected_color, mask_rect)
            assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, mask_rect)

    def test_to_surface__setsurface_narrower_and_taller_than_mask(self):
        """Ensures that setsurfaces narrower and taller than the mask work
        correctly.

        For this test the setsurface's width is less than the mask's width
        and the setsurface's height is greater than the mask's height.
        """
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        mask_size = (10, 8)
        narrow_tall_size = (6, 15)
        setsurface = pygame.Surface(narrow_tall_size, SRCALPHA, 32)
        setsurface_color = pygame.Color('red')
        setsurface.fill(setsurface_color)
        setsurface_rect = setsurface.get_rect()
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            to_surface = mask.to_surface(setsurface=setsurface)
            self.assertIsInstance(to_surface, pygame.Surface)
            self.assertEqual(to_surface.get_size(), mask_size)
            if fill:
                assertSurfaceFilled(self, to_surface, setsurface_color, setsurface_rect)
                assertSurfaceFilledIgnoreArea(self, to_surface, default_setcolor, setsurface_rect)
            else:
                assertSurfaceFilled(self, to_surface, default_unsetcolor)

    def test_to_surface__unsetsurface_narrower_and_taller_than_mask(self):
        """Ensures that unsetsurfaces narrower and taller than the mask work
        correctly.

        For this test the unsetsurface's width is less than the mask's width
        and the unsetsurface's height is greater than the mask's height.
        """
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        mask_size = (10, 8)
        narrow_tall_size = (6, 15)
        unsetsurface = pygame.Surface(narrow_tall_size, SRCALPHA, 32)
        unsetsurface_color = pygame.Color('red')
        unsetsurface.fill(unsetsurface_color)
        unsetsurface_rect = unsetsurface.get_rect()
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            to_surface = mask.to_surface(unsetsurface=unsetsurface)
            self.assertIsInstance(to_surface, pygame.Surface)
            self.assertEqual(to_surface.get_size(), mask_size)
            if fill:
                assertSurfaceFilled(self, to_surface, default_setcolor)
            else:
                assertSurfaceFilled(self, to_surface, unsetsurface_color, unsetsurface_rect)
                assertSurfaceFilledIgnoreArea(self, to_surface, default_unsetcolor, unsetsurface_rect)

    def test_to_surface__surface_narrower_and_shorter_than_mask(self):
        """Ensures that surfaces narrower and shorter than the mask work
        correctly.

        For this test the surface's width is less than the mask's width and
        the surface's height is less than the mask's height.
        """
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        mask_size = (10, 18)
        narrow_short_size = (6, 15)
        surface = pygame.Surface(narrow_short_size)
        surface_color = pygame.Color('red')
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            mask_rect = mask.get_rect()
            surface.fill(surface_color)
            expected_color = default_setcolor if fill else default_unsetcolor
            to_surface = mask.to_surface(surface)
            self.assertIs(to_surface, surface)
            self.assertEqual(to_surface.get_size(), narrow_short_size)
            assertSurfaceFilled(self, to_surface, expected_color, mask_rect)
            assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, mask_rect)

    def test_to_surface__setsurface_narrower_and_shorter_than_mask(self):
        """Ensures that setsurfaces narrower and shorter than the mask work
        correctly.

        For this test the setsurface's width is less than the mask's width
        and the setsurface's height is less than the mask's height.
        """
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        mask_size = (10, 18)
        narrow_short_size = (6, 15)
        setsurface = pygame.Surface(narrow_short_size, SRCALPHA, 32)
        setsurface_color = pygame.Color('red')
        setsurface.fill(setsurface_color)
        setsurface_rect = setsurface.get_rect()
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            to_surface = mask.to_surface(setsurface=setsurface)
            self.assertIsInstance(to_surface, pygame.Surface)
            self.assertEqual(to_surface.get_size(), mask_size)
            if fill:
                assertSurfaceFilled(self, to_surface, setsurface_color, setsurface_rect)
                assertSurfaceFilledIgnoreArea(self, to_surface, default_setcolor, setsurface_rect)
            else:
                assertSurfaceFilled(self, to_surface, default_unsetcolor)

    def test_to_surface__unsetsurface_narrower_and_shorter_than_mask(self):
        """Ensures that unsetsurfaces narrower and shorter than the mask work
        correctly.

        For this test the unsetsurface's width is less than the mask's width
        and the unsetsurface's height is less than the mask's height.
        """
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        mask_size = (10, 18)
        narrow_short_size = (6, 15)
        unsetsurface = pygame.Surface(narrow_short_size, SRCALPHA, 32)
        unsetsurface_color = pygame.Color('red')
        unsetsurface.fill(unsetsurface_color)
        unsetsurface_rect = unsetsurface.get_rect()
        for fill in (True, False):
            mask = pygame.mask.Mask(mask_size, fill=fill)
            to_surface = mask.to_surface(unsetsurface=unsetsurface)
            self.assertIsInstance(to_surface, pygame.Surface)
            self.assertEqual(to_surface.get_size(), mask_size)
            if fill:
                assertSurfaceFilled(self, to_surface, default_setcolor)
            else:
                assertSurfaceFilled(self, to_surface, unsetsurface_color, unsetsurface_rect)
                assertSurfaceFilledIgnoreArea(self, to_surface, default_unsetcolor, unsetsurface_rect)

    @unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
    def test_to_surface__all_surfaces_different_sizes_than_mask(self):
        """Ensures that all the surface parameters can be of different sizes."""
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        surface_color = pygame.Color('red')
        setsurface_color = pygame.Color('green')
        unsetsurface_color = pygame.Color('blue')
        mask_size = (10, 15)
        surface_size = (11, 14)
        setsurface_size = (9, 8)
        unsetsurface_size = (12, 16)
        surface = pygame.Surface(surface_size)
        setsurface = pygame.Surface(setsurface_size)
        unsetsurface = pygame.Surface(unsetsurface_size)
        surface.fill(surface_color)
        setsurface.fill(setsurface_color)
        unsetsurface.fill(unsetsurface_color)
        surface_rect = surface.get_rect()
        setsurface_rect = setsurface.get_rect()
        unsetsurface_rect = unsetsurface.get_rect()
        mask = pygame.mask.Mask(mask_size, fill=True)
        mask_rect = mask.get_rect()
        unfilled_rect = pygame.Rect((0, 0), (4, 5))
        unfilled_rect.center = mask_rect.center
        for pos in ((x, y) for x in range(unfilled_rect.x, unfilled_rect.w) for y in range(unfilled_rect.y, unfilled_rect.h)):
            mask.set_at(pos, 0)
        to_surface = mask.to_surface(surface, setsurface, unsetsurface)
        self.assertIs(to_surface, surface)
        self.assertEqual(to_surface.get_size(), surface_size)
        to_surface.lock()
        for pos in ((x, y) for x in range(surface_rect.w) for y in range(surface_rect.h)):
            if not mask_rect.collidepoint(pos):
                expected_color = surface_color
            elif mask.get_at(pos):
                if setsurface_rect.collidepoint(pos):
                    expected_color = setsurface_color
                else:
                    expected_color = default_setcolor
            elif unsetsurface_rect.collidepoint(pos):
                expected_color = unsetsurface_color
            else:
                expected_color = default_unsetcolor
            self.assertEqual(to_surface.get_at(pos), expected_color)
        to_surface.unlock()

    def test_to_surface__dest_locations(self):
        """Ensures dest values can be different locations on/off the surface."""
        SIDE = 7
        surface = pygame.Surface((SIDE, SIDE))
        surface_rect = surface.get_rect()
        dest_rect = surface_rect.copy()
        surface_color = pygame.Color('red')
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        directions = (((s, 0) for s in range(-SIDE, SIDE + 1)), ((0, s) for s in range(-SIDE, SIDE + 1)), ((s, s) for s in range(-SIDE, SIDE + 1)), ((-s, s) for s in range(-SIDE, SIDE + 1)))
        for fill in (True, False):
            mask = pygame.mask.Mask((SIDE, SIDE), fill=fill)
            expected_color = default_setcolor if fill else default_unsetcolor
            for direction in directions:
                for pos in direction:
                    dest_rect.topleft = pos
                    overlap_rect = dest_rect.clip(surface_rect)
                    surface.fill(surface_color)
                    to_surface = mask.to_surface(surface, dest=dest_rect)
                    assertSurfaceFilled(self, to_surface, expected_color, overlap_rect)
                    assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, overlap_rect)

    @unittest.expectedFailure
    @unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
    def test_to_surface__area_locations(self):
        """Ensures area rects can be different locations on/off the mask."""
        SIDE = 7
        surface = pygame.Surface((SIDE, SIDE))
        surface_color = pygame.Color('red')
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        directions = (((s, 0) for s in range(-SIDE, SIDE + 1)), ((0, s) for s in range(-SIDE, SIDE + 1)), ((s, s) for s in range(-SIDE, SIDE + 1)), ((-s, s) for s in range(-SIDE, SIDE + 1)))
        for fill in (True, False):
            mask = pygame.mask.Mask((SIDE, SIDE), fill=fill)
            mask_rect = mask.get_rect()
            area_rect = mask_rect.copy()
            expected_color = default_setcolor if fill else default_unsetcolor
            for direction in directions:
                for pos in direction:
                    area_rect.topleft = pos
                    overlap_rect = area_rect.clip(mask_rect)
                    overlap_rect.topleft = (0, 0)
                    surface.fill(surface_color)
                    to_surface = mask.to_surface(surface, area=area_rect)
                    assertSurfaceFilled(self, to_surface, expected_color, overlap_rect)
                    assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, overlap_rect)

    @unittest.expectedFailure
    def test_to_surface__dest_and_area_locations(self):
        """Ensures dest/area values can be different locations on/off the
        surface/mask.
        """
        SIDE = 5
        surface = pygame.Surface((SIDE, SIDE))
        surface_rect = surface.get_rect()
        dest_rect = surface_rect.copy()
        surface_color = pygame.Color('red')
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        dest_directions = (((s, 0) for s in range(-SIDE, SIDE + 1)), ((0, s) for s in range(-SIDE, SIDE + 1)), ((s, s) for s in range(-SIDE, SIDE + 1)), ((-s, s) for s in range(-SIDE, SIDE + 1)))
        area_positions = list(dest_directions[2])
        for fill in (True, False):
            mask = pygame.mask.Mask((SIDE, SIDE), fill=fill)
            mask_rect = mask.get_rect()
            area_rect = mask_rect.copy()
            expected_color = default_setcolor if fill else default_unsetcolor
            for dest_direction in dest_directions:
                for dest_pos in dest_direction:
                    dest_rect.topleft = dest_pos
                    for area_pos in area_positions:
                        area_rect.topleft = area_pos
                        area_overlap_rect = area_rect.clip(mask_rect)
                        area_overlap_rect.topleft = dest_rect.topleft
                        dest_overlap_rect = dest_rect.clip(area_overlap_rect)
                        surface.fill(surface_color)
                        to_surface = mask.to_surface(surface, dest=dest_rect, area=area_rect)
                        assertSurfaceFilled(self, to_surface, expected_color, dest_overlap_rect)
                        assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, dest_overlap_rect)

    @unittest.expectedFailure
    def test_to_surface__area_sizes(self):
        """Ensures area rects can be different sizes."""
        SIDE = 7
        SIZES = ((0, 0), (0, 1), (1, 0), (1, 1), (SIDE - 1, SIDE - 1), (SIDE - 1, SIDE), (SIDE, SIDE - 1), (SIDE, SIDE), (SIDE + 1, SIDE), (SIDE, SIDE + 1), (SIDE + 1, SIDE + 1))
        surface = pygame.Surface((SIDE, SIDE))
        surface_color = pygame.Color('red')
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        for fill in (True, False):
            mask = pygame.mask.Mask((SIDE, SIDE), fill=fill)
            mask_rect = mask.get_rect()
            expected_color = default_setcolor if fill else default_unsetcolor
            for size in SIZES:
                area_rect = pygame.Rect((0, 0), size)
                for pos in self.ORIGIN_OFFSETS:
                    area_rect.topleft = pos
                    overlap_rect = area_rect.clip(mask_rect)
                    overlap_rect.topleft = (0, 0)
                    surface.fill(surface_color)
                    to_surface = mask.to_surface(surface, area=area_rect)
                    assertSurfaceFilled(self, to_surface, expected_color, overlap_rect)
                    assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, overlap_rect)

    def test_to_surface__surface_color_alphas(self):
        """Ensures the setsurface/unsetsurface color alpha values are respected."""
        size = (13, 17)
        setsurface_color = pygame.Color('green')
        setsurface_color.a = 53
        unsetsurface_color = pygame.Color('blue')
        unsetsurface_color.a = 109
        setsurface = pygame.Surface(size, flags=SRCALPHA, depth=32)
        unsetsurface = pygame.Surface(size, flags=SRCALPHA, depth=32)
        setsurface.fill(setsurface_color)
        unsetsurface.fill(unsetsurface_color)
        for fill in (True, False):
            mask = pygame.mask.Mask(size, fill=fill)
            expected_color = setsurface_color if fill else unsetsurface_color
            to_surface = mask.to_surface(setsurface=setsurface, unsetsurface=unsetsurface)
            self.assertIsInstance(to_surface, pygame.Surface)
            self.assertEqual(to_surface.get_size(), size)
            assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__color_alphas(self):
        """Ensures the setcolor/unsetcolor alpha values are respected."""
        size = (13, 17)
        setcolor = pygame.Color('green')
        setcolor.a = 35
        unsetcolor = pygame.Color('blue')
        unsetcolor.a = 213
        for fill in (True, False):
            mask = pygame.mask.Mask(size, fill=fill)
            expected_color = setcolor if fill else unsetcolor
            to_surface = mask.to_surface(setcolor=setcolor, unsetcolor=unsetcolor)
            self.assertIsInstance(to_surface, pygame.Surface)
            self.assertEqual(to_surface.get_size(), size)
            assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__depths(self):
        """Ensures to_surface works correctly with supported surface depths."""
        size = (13, 17)
        surface_color = pygame.Color('red')
        setsurface_color = pygame.Color('green')
        unsetsurface_color = pygame.Color('blue')
        for depth in (8, 16, 24, 32):
            surface = pygame.Surface(size, depth=depth)
            setsurface = pygame.Surface(size, depth=depth)
            unsetsurface = pygame.Surface(size, depth=depth)
            surface.fill(surface_color)
            setsurface.fill(setsurface_color)
            unsetsurface.fill(unsetsurface_color)
            for fill in (True, False):
                mask = pygame.mask.Mask(size, fill=fill)
                expected_color = setsurface.get_at((0, 0)) if fill else unsetsurface.get_at((0, 0))
                to_surface = mask.to_surface(surface, setsurface, unsetsurface)
                self.assertIsInstance(to_surface, pygame.Surface)
                self.assertEqual(to_surface.get_size(), size)
                assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__different_depths(self):
        """Ensures an exception is raised when surfaces have different depths."""
        size = (13, 17)
        surface_color = pygame.Color('red')
        setsurface_color = pygame.Color('green')
        unsetsurface_color = pygame.Color('blue')
        mask = pygame.mask.Mask(size)
        test_depths = ((8, 8, 16), (8, 8, 24), (8, 8, 32), (16, 16, 24), (16, 16, 32), (24, 16, 8), (32, 16, 16), (32, 32, 16), (32, 24, 32))
        for depths in test_depths:
            surface = pygame.Surface(size, depth=depths[0])
            setsurface = pygame.Surface(size, depth=depths[1])
            unsetsurface = pygame.Surface(size, depth=depths[2])
            surface.fill(surface_color)
            setsurface.fill(setsurface_color)
            unsetsurface.fill(unsetsurface_color)
            with self.assertRaises(ValueError):
                mask.to_surface(surface, setsurface, unsetsurface)

    def test_to_surface__different_depths_with_created_surfaces(self):
        """Ensures an exception is raised when surfaces have different depths
        than the created surface.
        """
        size = (13, 17)
        setsurface_color = pygame.Color('green')
        unsetsurface_color = pygame.Color('blue')
        mask = pygame.mask.Mask(size)
        test_depths = ((8, 8), (16, 16), (24, 24), (24, 16), (32, 8), (32, 16), (32, 24), (16, 32))
        for set_depth, unset_depth in test_depths:
            setsurface = pygame.Surface(size, depth=set_depth)
            unsetsurface = pygame.Surface(size, depth=unset_depth)
            setsurface.fill(setsurface_color)
            unsetsurface.fill(unsetsurface_color)
            with self.assertRaises(ValueError):
                mask.to_surface(setsurface=setsurface, unsetsurface=unsetsurface)

    def test_to_surface__same_srcalphas(self):
        """Ensures to_surface works correctly when the SRCALPHA flag is set or not."""
        size = (13, 17)
        surface_color = pygame.Color('red')
        setsurface_color = pygame.Color('green')
        unsetsurface_color = pygame.Color('blue')
        for depth in (16, 32):
            for flags in (0, SRCALPHA):
                surface = pygame.Surface(size, flags=flags, depth=depth)
                setsurface = pygame.Surface(size, flags=flags, depth=depth)
                unsetsurface = pygame.Surface(size, flags=flags, depth=depth)
                surface.fill(surface_color)
                setsurface.fill(setsurface_color)
                unsetsurface.fill(unsetsurface_color)
                for fill in (True, False):
                    mask = pygame.mask.Mask(size, fill=fill)
                    expected_color = setsurface_color if fill else unsetsurface_color
                    to_surface = mask.to_surface(surface, setsurface, unsetsurface)
                    self.assertIsInstance(to_surface, pygame.Surface)
                    self.assertEqual(to_surface.get_size(), size)
                    assertSurfaceFilled(self, to_surface, expected_color)
                    if flags:
                        self.assertTrue(to_surface.get_flags() & flags)

    def test_to_surface__same_srcalphas_with_created_surfaces(self):
        """Ensures to_surface works correctly when it creates a surface
        and the SRCALPHA flag is set on both setsurface and unsetsurface.
        """
        size = (13, 17)
        setsurface_color = pygame.Color('green')
        unsetsurface_color = pygame.Color('blue')
        expected_flags = SRCALPHA
        setsurface = pygame.Surface(size, flags=expected_flags, depth=32)
        unsetsurface = pygame.Surface(size, flags=expected_flags, depth=32)
        setsurface.fill(setsurface_color)
        unsetsurface.fill(unsetsurface_color)
        for fill in (True, False):
            mask = pygame.mask.Mask(size, fill=fill)
            expected_color = setsurface_color if fill else unsetsurface_color
            to_surface = mask.to_surface(setsurface=setsurface, unsetsurface=unsetsurface)
            self.assertIsInstance(to_surface, pygame.Surface)
            self.assertEqual(to_surface.get_size(), size)
            assertSurfaceFilled(self, to_surface, expected_color)
            self.assertTrue(to_surface.get_flags() & expected_flags)

    def test_to_surface__different_srcalphas(self):
        """Ensures an exception is raised when surfaces have different SRCALPHA
        flag settings.
        """
        size = (13, 17)
        surface_color = pygame.Color('red')
        setsurface_color = pygame.Color('green')
        unsetsurface_color = pygame.Color('blue')
        mask = pygame.mask.Mask(size)
        test_flags = ((SRCALPHA, 0, 0), (SRCALPHA, SRCALPHA, 0), (0, SRCALPHA, SRCALPHA), (0, 0, SRCALPHA))
        for depth in (16, 32):
            for flags in test_flags:
                surface = pygame.Surface(size, flags=flags[0], depth=depth)
                setsurface = pygame.Surface(size, flags=flags[1], depth=depth)
                unsetsurface = pygame.Surface(size, flags=flags[2], depth=depth)
                surface.fill(surface_color)
                setsurface.fill(setsurface_color)
                unsetsurface.fill(unsetsurface_color)
                with self.assertRaises(ValueError):
                    mask.to_surface(surface, setsurface, unsetsurface)

    def test_to_surface__different_srcalphas_with_created_surfaces(self):
        """Ensures an exception is raised when surfaces have different SRCALPHA
        flag settings than the created surface.
        """
        size = (13, 17)
        setsurface_color = pygame.Color('green')
        unsetsurface_color = pygame.Color('blue')
        mask = pygame.mask.Mask(size)
        for depth in (16, 32):
            for flags in ((0, 0), (SRCALPHA, 0), (0, SRCALPHA)):
                setsurface = pygame.Surface(size, flags=flags[0], depth=depth)
                unsetsurface = pygame.Surface(size, flags=flags[1], depth=depth)
                setsurface.fill(setsurface_color)
                unsetsurface.fill(unsetsurface_color)
                with self.assertRaises(ValueError):
                    mask.to_surface(setsurface=setsurface, unsetsurface=unsetsurface)

    def test_to_surface__dest_on_surface(self):
        """Ensures dest values on the surface work correctly
        when using the defaults for setcolor and unsetcolor.
        """
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        width, height = size = (5, 9)
        surface = pygame.Surface(size, SRCALPHA, 32)
        surface_color = pygame.Color('red')
        for fill in (True, False):
            mask = pygame.mask.Mask(size, fill=fill)
            mask_rect = mask.get_rect()
            expected_color = default_setcolor if fill else default_unsetcolor
            for dest in ((x, y) for y in range(height) for x in range(width)):
                surface.fill(surface_color)
                mask_rect.topleft = dest
                to_surface = mask.to_surface(surface, dest=dest)
                self.assertIs(to_surface, surface)
                self.assertEqual(to_surface.get_size(), size)
                assertSurfaceFilled(self, to_surface, expected_color, mask_rect)
                assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, mask_rect)

    def test_to_surface__dest_on_surface_with_setsurface_unsetsurface(self):
        """Ensures dest values on the surface work correctly
        when using setsurface and unsetsurface.
        """
        width, height = size = (5, 9)
        surface = pygame.Surface(size, SRCALPHA, 32)
        surface_color = pygame.Color('red')
        setsurface = surface.copy()
        setsurface_color = pygame.Color('green')
        setsurface.fill(setsurface_color)
        unsetsurface = surface.copy()
        unsetsurface_color = pygame.Color('blue')
        unsetsurface.fill(unsetsurface_color)
        kwargs = {'surface': surface, 'setsurface': setsurface, 'unsetsurface': unsetsurface, 'dest': None}
        color_kwargs = dict(kwargs)
        color_kwargs.update((('setcolor', None), ('unsetcolor', None)))
        for fill in (True, False):
            mask = pygame.mask.Mask(size, fill=fill)
            mask_rect = mask.get_rect()
            expected_color = setsurface_color if fill else unsetsurface_color
            for dest in ((x, y) for y in range(height) for x in range(width)):
                mask_rect.topleft = dest
                for use_color_params in (True, False):
                    surface.fill(surface_color)
                    test_kwargs = color_kwargs if use_color_params else kwargs
                    test_kwargs['dest'] = dest
                    to_surface = mask.to_surface(**test_kwargs)
                    self.assertIs(to_surface, surface)
                    self.assertEqual(to_surface.get_size(), size)
                    assertSurfaceFilled(self, to_surface, expected_color, mask_rect)
                    assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, mask_rect)

    def test_to_surface__dest_off_surface(self):
        """Ensures dest values off the surface work correctly
        when using the defaults for setcolor and unsetcolor.
        """
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        width, height = size = (5, 7)
        surface = pygame.Surface(size, SRCALPHA, 32)
        surface_color = pygame.Color('red')
        dests = [(-width, -height), (-width, 0), (0, -height)]
        dests.extend(off_corners(surface.get_rect()))
        for fill in (True, False):
            mask = pygame.mask.Mask(size, fill=fill)
            mask_rect = mask.get_rect()
            expected_color = default_setcolor if fill else default_unsetcolor
            for dest in dests:
                surface.fill(surface_color)
                mask_rect.topleft = dest
                to_surface = mask.to_surface(surface, dest=dest)
                self.assertIs(to_surface, surface)
                self.assertEqual(to_surface.get_size(), size)
                assertSurfaceFilled(self, to_surface, expected_color, mask_rect)
                assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, mask_rect)

    def test_to_surface__dest_off_surface_with_setsurface_unsetsurface(self):
        """Ensures dest values off the surface work correctly
        when using setsurface and unsetsurface.
        """
        width, height = size = (5, 7)
        surface = pygame.Surface(size, SRCALPHA, 32)
        surface_color = pygame.Color('red')
        setsurface = surface.copy()
        setsurface_color = pygame.Color('green')
        setsurface.fill(setsurface_color)
        unsetsurface = surface.copy()
        unsetsurface_color = pygame.Color('blue')
        unsetsurface.fill(unsetsurface_color)
        dests = [(-width, -height), (-width, 0), (0, -height)]
        dests.extend(off_corners(surface.get_rect()))
        kwargs = {'surface': surface, 'setsurface': setsurface, 'unsetsurface': unsetsurface, 'dest': None}
        color_kwargs = dict(kwargs)
        color_kwargs.update((('setcolor', None), ('unsetcolor', None)))
        for fill in (True, False):
            mask = pygame.mask.Mask(size, fill=fill)
            mask_rect = mask.get_rect()
            expected_color = setsurface_color if fill else unsetsurface_color
            for dest in dests:
                mask_rect.topleft = dest
                for use_color_params in (True, False):
                    surface.fill(surface_color)
                    test_kwargs = color_kwargs if use_color_params else kwargs
                    test_kwargs['dest'] = dest
                    to_surface = mask.to_surface(**test_kwargs)
                    self.assertIs(to_surface, surface)
                    self.assertEqual(to_surface.get_size(), size)
                    assertSurfaceFilled(self, to_surface, expected_color, mask_rect)
                    assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, mask_rect)

    @unittest.expectedFailure
    @unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
    def test_to_surface__area_on_mask(self):
        """Ensures area values on the mask work correctly
        when using the defaults for setcolor and unsetcolor.
        """
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        width, height = size = (5, 9)
        surface = pygame.Surface(size, SRCALPHA, 32)
        surface_color = pygame.Color('red')
        for fill in (True, False):
            mask = pygame.mask.Mask(size, fill=fill)
            mask_rect = mask.get_rect()
            area_rect = mask_rect.copy()
            expected_color = default_setcolor if fill else default_unsetcolor
            for pos in ((x, y) for y in range(height) for x in range(width)):
                surface.fill(surface_color)
                area_rect.topleft = pos
                overlap_rect = mask_rect.clip(area_rect)
                overlap_rect.topleft = (0, 0)
                to_surface = mask.to_surface(surface, area=area_rect)
                self.assertIs(to_surface, surface)
                self.assertEqual(to_surface.get_size(), size)
                assertSurfaceFilled(self, to_surface, expected_color, overlap_rect)
                assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, overlap_rect)

    @unittest.expectedFailure
    def test_to_surface__area_on_mask_with_setsurface_unsetsurface(self):
        """Ensures area values on the mask work correctly
        when using setsurface and unsetsurface.
        """
        width, height = size = (5, 9)
        surface = pygame.Surface(size, SRCALPHA, 32)
        surface_color = pygame.Color('red')
        setsurface = surface.copy()
        setsurface_color = pygame.Color('green')
        setsurface.fill(setsurface_color)
        unsetsurface = surface.copy()
        unsetsurface_color = pygame.Color('blue')
        unsetsurface.fill(unsetsurface_color)
        kwargs = {'surface': surface, 'setsurface': setsurface, 'unsetsurface': unsetsurface, 'area': pygame.Rect((0, 0), size)}
        color_kwargs = dict(kwargs)
        color_kwargs.update((('setcolor', None), ('unsetcolor', None)))
        for fill in (True, False):
            mask = pygame.mask.Mask(size, fill=fill)
            mask_rect = mask.get_rect()
            area_rect = mask_rect.copy()
            expected_color = setsurface_color if fill else unsetsurface_color
            for pos in ((x, y) for y in range(height) for x in range(width)):
                area_rect.topleft = pos
                overlap_rect = mask_rect.clip(area_rect)
                overlap_rect.topleft = (0, 0)
                for use_color_params in (True, False):
                    surface.fill(surface_color)
                    test_kwargs = color_kwargs if use_color_params else kwargs
                    test_kwargs['area'].topleft = pos
                    overlap_rect = mask_rect.clip(test_kwargs['area'])
                    overlap_rect.topleft = (0, 0)
                    to_surface = mask.to_surface(**test_kwargs)
                    self.assertIs(to_surface, surface)
                    self.assertEqual(to_surface.get_size(), size)
                    assertSurfaceFilled(self, to_surface, expected_color, overlap_rect)
                    assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, overlap_rect)

    @unittest.expectedFailure
    @unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
    def test_to_surface__area_off_mask(self):
        """Ensures area values off the mask work correctly
        when using the defaults for setcolor and unsetcolor.
        """
        default_setcolor = pygame.Color('white')
        default_unsetcolor = pygame.Color('black')
        width, height = size = (5, 7)
        surface = pygame.Surface(size, SRCALPHA, 32)
        surface_color = pygame.Color('red')
        positions = [(-width, -height), (-width, 0), (0, -height)]
        positions.extend(off_corners(pygame.Rect((0, 0), (width, height))))
        for fill in (True, False):
            mask = pygame.mask.Mask(size, fill=fill)
            mask_rect = mask.get_rect()
            area_rect = mask_rect.copy()
            expected_color = default_setcolor if fill else default_unsetcolor
            for pos in positions:
                surface.fill(surface_color)
                area_rect.topleft = pos
                overlap_rect = mask_rect.clip(area_rect)
                overlap_rect.topleft = (0, 0)
                to_surface = mask.to_surface(surface, area=area_rect)
                self.assertIs(to_surface, surface)
                self.assertEqual(to_surface.get_size(), size)
                assertSurfaceFilled(self, to_surface, expected_color, overlap_rect)
                assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, overlap_rect)

    @unittest.expectedFailure
    @unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
    def test_to_surface__area_off_mask_with_setsurface_unsetsurface(self):
        """Ensures area values off the mask work correctly
        when using setsurface and unsetsurface.
        """
        width, height = size = (5, 7)
        surface = pygame.Surface(size, SRCALPHA, 32)
        surface_color = pygame.Color('red')
        setsurface = surface.copy()
        setsurface_color = pygame.Color('green')
        setsurface.fill(setsurface_color)
        unsetsurface = surface.copy()
        unsetsurface_color = pygame.Color('blue')
        unsetsurface.fill(unsetsurface_color)
        positions = [(-width, -height), (-width, 0), (0, -height)]
        positions.extend(off_corners(pygame.Rect((0, 0), (width, height))))
        kwargs = {'surface': surface, 'setsurface': setsurface, 'unsetsurface': unsetsurface, 'area': pygame.Rect((0, 0), size)}
        color_kwargs = dict(kwargs)
        color_kwargs.update((('setcolor', None), ('unsetcolor', None)))
        for fill in (True, False):
            mask = pygame.mask.Mask(size, fill=fill)
            mask_rect = mask.get_rect()
            expected_color = setsurface_color if fill else unsetsurface_color
            for pos in positions:
                for use_color_params in (True, False):
                    surface.fill(surface_color)
                    test_kwargs = color_kwargs if use_color_params else kwargs
                    test_kwargs['area'].topleft = pos
                    overlap_rect = mask_rect.clip(test_kwargs['area'])
                    overlap_rect.topleft = (0, 0)
                    to_surface = mask.to_surface(**test_kwargs)
                    self.assertIs(to_surface, surface)
                    self.assertEqual(to_surface.get_size(), size)
                    assertSurfaceFilled(self, to_surface, expected_color, overlap_rect)
                    assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, overlap_rect)

    def test_to_surface__surface_with_zero_size(self):
        """Ensures zero sized surfaces are handled correctly."""
        expected_ref_count = 3
        size = (0, 0)
        surface = pygame.Surface(size)
        mask = pygame.mask.Mask((3, 4), fill=True)
        to_surface = mask.to_surface(surface)
        self.assertIs(to_surface, surface)
        if not IS_PYPY:
            self.assertEqual(sys.getrefcount(to_surface), expected_ref_count)
        self.assertEqual(to_surface.get_size(), size)

    def test_to_surface__setsurface_with_zero_size(self):
        """Ensures zero sized setsurfaces are handled correctly."""
        expected_ref_count = 2
        expected_flag = SRCALPHA
        expected_depth = 32
        expected_color = pygame.Color('white')
        mask_size = (2, 4)
        mask = pygame.mask.Mask(mask_size, fill=True)
        setsurface = pygame.Surface((0, 0), expected_flag, expected_depth)
        to_surface = mask.to_surface(setsurface=setsurface)
        self.assertIsInstance(to_surface, pygame.Surface)
        if not IS_PYPY:
            self.assertEqual(sys.getrefcount(to_surface), expected_ref_count)
        self.assertTrue(to_surface.get_flags() & expected_flag)
        self.assertEqual(to_surface.get_bitsize(), expected_depth)
        self.assertEqual(to_surface.get_size(), mask_size)
        assertSurfaceFilled(self, to_surface, expected_color)

    def test_to_surface__unsetsurface_with_zero_size(self):
        """Ensures zero sized unsetsurfaces are handled correctly."""
        expected_ref_count = 2
        expected_flag = SRCALPHA
        expected_depth = 32
        expected_color = pygame.Color('black')
        mask_size = (4, 2)
        mask = pygame.mask.Mask(mask_size)
        unsetsurface = pygame.Surface((0, 0), expected_flag, expected_depth)
        to_surface = mask.to_surface(unsetsurface=unsetsurface)
        self.assertIsInstance(to_surface, pygame.Surface)
        if not IS_PYPY:
            self.assertEqual(sys.getrefcount(to_surface), expected_ref_count)
        self.assertTrue(to_surface.get_flags() & expected_flag)
        self.assertEqual(to_surface.get_bitsize(), expected_depth)
        self.assertEqual(to_surface.get_size(), mask_size)
        assertSurfaceFilled(self, to_surface, expected_color)

    def test_zero_mask(self):
        """Ensures masks can be created with zero sizes."""
        for size in ((100, 0), (0, 100), (0, 0)):
            for fill in (True, False):
                msg = f'size={size}, fill={fill}'
                mask = pygame.mask.Mask(size, fill=fill)
                self.assertIsInstance(mask, pygame.mask.Mask, msg)
                self.assertEqual(mask.get_size(), size, msg)

    def test_zero_mask_copy(self):
        """Ensures copy correctly handles zero sized masks."""
        for expected_size in ((11, 0), (0, 11), (0, 0)):
            mask = pygame.mask.Mask(expected_size)
            mask_copy = mask.copy()
            self.assertIsInstance(mask_copy, pygame.mask.Mask)
            self.assertIsNot(mask_copy, mask)
            assertMaskEqual(self, mask_copy, mask)

    def test_zero_mask_get_size(self):
        """Ensures get_size correctly handles zero sized masks."""
        for expected_size in ((41, 0), (0, 40), (0, 0)):
            mask = pygame.mask.Mask(expected_size)
            size = mask.get_size()
            self.assertEqual(size, expected_size)

    def test_zero_mask_get_rect(self):
        """Ensures get_rect correctly handles zero sized masks."""
        for expected_size in ((4, 0), (0, 4), (0, 0)):
            expected_rect = pygame.Rect((0, 0), expected_size)
            mask = pygame.mask.Mask(expected_size)
            rect = mask.get_rect()
            self.assertEqual(rect, expected_rect)

    def test_zero_mask_get_at(self):
        """Ensures get_at correctly handles zero sized masks."""
        for size in ((51, 0), (0, 50), (0, 0)):
            mask = pygame.mask.Mask(size)
            with self.assertRaises(IndexError):
                value = mask.get_at((0, 0))

    def test_zero_mask_set_at(self):
        """Ensures set_at correctly handles zero sized masks."""
        for size in ((31, 0), (0, 30), (0, 0)):
            mask = pygame.mask.Mask(size)
            with self.assertRaises(IndexError):
                mask.set_at((0, 0))

    def test_zero_mask_overlap(self):
        """Ensures overlap correctly handles zero sized masks.

        Tests combinations of sized and zero sized masks.
        """
        offset = (0, 0)
        for size1, size2 in zero_size_pairs(51, 42):
            msg = f'size1={size1}, size2={size2}'
            mask1 = pygame.mask.Mask(size1, fill=True)
            mask2 = pygame.mask.Mask(size2, fill=True)
            overlap_pos = mask1.overlap(mask2, offset)
            self.assertIsNone(overlap_pos, msg)

    def test_zero_mask_overlap_area(self):
        """Ensures overlap_area correctly handles zero sized masks.

        Tests combinations of sized and zero sized masks.
        """
        offset = (0, 0)
        expected_count = 0
        for size1, size2 in zero_size_pairs(41, 52):
            msg = f'size1={size1}, size2={size2}'
            mask1 = pygame.mask.Mask(size1, fill=True)
            mask2 = pygame.mask.Mask(size2, fill=True)
            overlap_count = mask1.overlap_area(mask2, offset)
            self.assertEqual(overlap_count, expected_count, msg)

    def test_zero_mask_overlap_mask(self):
        """Ensures overlap_mask correctly handles zero sized masks.

        Tests combinations of sized and zero sized masks.
        """
        offset = (0, 0)
        expected_count = 0
        for size1, size2 in zero_size_pairs(43, 53):
            msg = f'size1={size1}, size2={size2}'
            mask1 = pygame.mask.Mask(size1, fill=True)
            mask2 = pygame.mask.Mask(size2, fill=True)
            overlap_mask = mask1.overlap_mask(mask2, offset)
            self.assertIsInstance(overlap_mask, pygame.mask.Mask, msg)
            self.assertEqual(overlap_mask.count(), expected_count, msg)
            self.assertEqual(overlap_mask.get_size(), size1, msg)

    def test_zero_mask_fill(self):
        """Ensures fill correctly handles zero sized masks."""
        expected_count = 0
        for size in ((100, 0), (0, 100), (0, 0)):
            mask = pygame.mask.Mask(size)
            mask.fill()
            self.assertEqual(mask.count(), expected_count, f'size={size}')

    def test_zero_mask_clear(self):
        sizes = ((100, 0), (0, 100), (0, 0))
        for size in sizes:
            mask = pygame.mask.Mask(size)
            mask.clear()
            self.assertEqual(mask.count(), 0)

    def test_zero_mask_flip(self):
        sizes = ((100, 0), (0, 100), (0, 0))
        for size in sizes:
            mask = pygame.mask.Mask(size)
            mask.invert()
            self.assertEqual(mask.count(), 0)

    def test_zero_mask_scale(self):
        sizes = ((100, 0), (0, 100), (0, 0))
        for size in sizes:
            mask = pygame.mask.Mask(size)
            mask2 = mask.scale((2, 3))
            self.assertIsInstance(mask2, pygame.mask.Mask)
            self.assertEqual(mask2.get_size(), (2, 3))

    def test_zero_mask_draw(self):
        """Ensures draw correctly handles zero sized masks.

        Tests combinations of sized and zero sized masks.
        """
        offset = (0, 0)
        for size1, size2 in zero_size_pairs(31, 37):
            msg = f'size1={size1}, size2={size2}'
            mask1 = pygame.mask.Mask(size1, fill=True)
            mask2 = pygame.mask.Mask(size2, fill=True)
            expected_count = mask1.count()
            mask1.draw(mask2, offset)
            self.assertEqual(mask1.count(), expected_count, msg)
            self.assertEqual(mask1.get_size(), size1, msg)

    def test_zero_mask_erase(self):
        """Ensures erase correctly handles zero sized masks.

        Tests combinations of sized and zero sized masks.
        """
        offset = (0, 0)
        for size1, size2 in zero_size_pairs(29, 23):
            msg = f'size1={size1}, size2={size2}'
            mask1 = pygame.mask.Mask(size1, fill=True)
            mask2 = pygame.mask.Mask(size2, fill=True)
            expected_count = mask1.count()
            mask1.erase(mask2, offset)
            self.assertEqual(mask1.count(), expected_count, msg)
            self.assertEqual(mask1.get_size(), size1, msg)

    def test_zero_mask_count(self):
        sizes = ((100, 0), (0, 100), (0, 0))
        for size in sizes:
            mask = pygame.mask.Mask(size, fill=True)
            self.assertEqual(mask.count(), 0)

    def test_zero_mask_centroid(self):
        sizes = ((100, 0), (0, 100), (0, 0))
        for size in sizes:
            mask = pygame.mask.Mask(size)
            self.assertEqual(mask.centroid(), (0, 0))

    def test_zero_mask_angle(self):
        sizes = ((100, 0), (0, 100), (0, 0))
        for size in sizes:
            mask = pygame.mask.Mask(size)
            self.assertEqual(mask.angle(), 0.0)

    def test_zero_mask_outline(self):
        """Ensures outline correctly handles zero sized masks."""
        expected_points = []
        for size in ((61, 0), (0, 60), (0, 0)):
            mask = pygame.mask.Mask(size)
            points = mask.outline()
            self.assertListEqual(points, expected_points, f'size={size}')

    def test_zero_mask_outline__with_arg(self):
        """Ensures outline correctly handles zero sized masks
        when using the skip pixels argument."""
        expected_points = []
        for size in ((66, 0), (0, 65), (0, 0)):
            mask = pygame.mask.Mask(size)
            points = mask.outline(10)
            self.assertListEqual(points, expected_points, f'size={size}')

    def test_zero_mask_convolve(self):
        """Ensures convolve correctly handles zero sized masks.

        Tests the different combinations of sized and zero sized masks.
        """
        for size1 in ((17, 13), (71, 0), (0, 70), (0, 0)):
            mask1 = pygame.mask.Mask(size1, fill=True)
            for size2 in ((11, 7), (81, 0), (0, 60), (0, 0)):
                msg = f'sizes={size1}, {size2}'
                mask2 = pygame.mask.Mask(size2, fill=True)
                expected_size = (max(0, size1[0] + size2[0] - 1), max(0, size1[1] + size2[1] - 1))
                mask = mask1.convolve(mask2)
                self.assertIsInstance(mask, pygame.mask.Mask, msg)
                self.assertIsNot(mask, mask2, msg)
                self.assertEqual(mask.get_size(), expected_size, msg)

    def test_zero_mask_convolve__with_output_mask(self):
        """Ensures convolve correctly handles zero sized masks
        when using an output mask argument.

        Tests the different combinations of sized and zero sized masks.
        """
        for size1 in ((11, 17), (91, 0), (0, 90), (0, 0)):
            mask1 = pygame.mask.Mask(size1, fill=True)
            for size2 in ((13, 11), (83, 0), (0, 62), (0, 0)):
                mask2 = pygame.mask.Mask(size2, fill=True)
                for output_size in ((7, 5), (71, 0), (0, 70), (0, 0)):
                    msg = f'sizes={size1}, {size2}, {output_size}'
                    output_mask = pygame.mask.Mask(output_size)
                    mask = mask1.convolve(mask2, output_mask)
                    self.assertIsInstance(mask, pygame.mask.Mask, msg)
                    self.assertIs(mask, output_mask, msg)
                    self.assertEqual(mask.get_size(), output_size, msg)

    def test_zero_mask_connected_component(self):
        """Ensures connected_component correctly handles zero sized masks."""
        expected_count = 0
        for size in ((81, 0), (0, 80), (0, 0)):
            msg = f'size={size}'
            mask = pygame.mask.Mask(size)
            cc_mask = mask.connected_component()
            self.assertIsInstance(cc_mask, pygame.mask.Mask, msg)
            self.assertEqual(cc_mask.get_size(), size)
            self.assertEqual(cc_mask.count(), expected_count, msg)

    def test_zero_mask_connected_component__indexed(self):
        """Ensures connected_component correctly handles zero sized masks
        when using an index argument."""
        for size in ((91, 0), (0, 90), (0, 0)):
            mask = pygame.mask.Mask(size)
            with self.assertRaises(IndexError):
                cc_mask = mask.connected_component((0, 0))

    def test_zero_mask_connected_components(self):
        """Ensures connected_components correctly handles zero sized masks."""
        expected_cc_masks = []
        for size in ((11, 0), (0, 10), (0, 0)):
            mask = pygame.mask.Mask(size)
            cc_masks = mask.connected_components()
            self.assertListEqual(cc_masks, expected_cc_masks, f'size={size}')

    def test_zero_mask_get_bounding_rects(self):
        """Ensures get_bounding_rects correctly handles zero sized masks."""
        expected_bounding_rects = []
        for size in ((21, 0), (0, 20), (0, 0)):
            mask = pygame.mask.Mask(size)
            bounding_rects = mask.get_bounding_rects()
            self.assertListEqual(bounding_rects, expected_bounding_rects, f'size={size}')

    def test_zero_mask_to_surface(self):
        """Ensures to_surface correctly handles zero sized masks and surfaces."""
        mask_color = pygame.Color('blue')
        surf_color = pygame.Color('red')
        for surf_size in ((7, 3), (7, 0), (0, 7), (0, 0)):
            surface = pygame.Surface(surf_size, SRCALPHA, 32)
            surface.fill(surf_color)
            for mask_size in ((5, 0), (0, 5), (0, 0)):
                mask = pygame.mask.Mask(mask_size, fill=True)
                to_surface = mask.to_surface(surface, setcolor=mask_color)
                self.assertIs(to_surface, surface)
                self.assertEqual(to_surface.get_size(), surf_size)
                if 0 not in surf_size:
                    assertSurfaceFilled(self, to_surface, surf_color)

    def test_zero_mask_to_surface__create_surface(self):
        """Ensures to_surface correctly handles zero sized masks and surfaces
        when it has to create a default surface.
        """
        mask_color = pygame.Color('blue')
        for mask_size in ((3, 0), (0, 3), (0, 0)):
            mask = pygame.mask.Mask(mask_size, fill=True)
            to_surface = mask.to_surface(setcolor=mask_color)
            self.assertIsInstance(to_surface, pygame.Surface)
            self.assertEqual(to_surface.get_size(), mask_size)