from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
@unittest.skipIf(IS_PYPY, 'pypy has lots of mask failures')
class MaskModuleTest(unittest.TestCase):

    def test_from_surface(self):
        """Ensures from_surface creates a mask with the correct bits set.

        This test checks the masks created by the from_surface function using
        16 and 32 bit surfaces. Each alpha value (0-255) is tested against
        several different threshold values.
        Note: On 16 bit surface the requested alpha value can differ from what
              is actually set. This test uses the value read from the surface.
        """
        threshold_count = 256
        surface_color = [55, 155, 255, 0]
        expected_size = (11, 9)
        all_set_count = expected_size[0] * expected_size[1]
        none_set_count = 0
        for depth in (16, 32):
            surface = pygame.Surface(expected_size, SRCALPHA, depth)
            for alpha in range(threshold_count):
                surface_color[3] = alpha
                surface.fill(surface_color)
                if depth < 32:
                    alpha = surface.get_at((0, 0))[3]
                threshold_test_values = {-1, 0, alpha - 1, alpha, alpha + 1, 255, 256}
                for threshold in threshold_test_values:
                    msg = f'depth={depth}, alpha={alpha}, threshold={threshold}'
                    if alpha > threshold:
                        expected_count = all_set_count
                    else:
                        expected_count = none_set_count
                    mask = pygame.mask.from_surface(surface=surface, threshold=threshold)
                    self.assertIsInstance(mask, pygame.mask.Mask, msg)
                    self.assertEqual(mask.get_size(), expected_size, msg)
                    self.assertEqual(mask.count(), expected_count, msg)

    def test_from_surface__different_alphas_32bit(self):
        """Ensures from_surface creates a mask with the correct bits set
        when pixels have different alpha values (32 bits surfaces).

        This test checks the masks created by the from_surface function using
        a 32 bit surface. The surface is created with each pixel having a
        different alpha value (0-255). This surface is tested over a range
        of threshold values (0-255).
        """
        offset = (0, 0)
        threshold_count = 256
        surface_color = [10, 20, 30, 0]
        expected_size = (threshold_count, 1)
        expected_mask = pygame.Mask(expected_size, fill=True)
        surface = pygame.Surface(expected_size, SRCALPHA, 32)
        surface.lock()
        for a in range(threshold_count):
            surface_color[3] = a
            surface.set_at((a, 0), surface_color)
        surface.unlock()
        for threshold in range(threshold_count):
            msg = f'threshold={threshold}'
            expected_mask.set_at((threshold, 0), 0)
            expected_count = expected_mask.count()
            mask = pygame.mask.from_surface(surface, threshold)
            self.assertIsInstance(mask, pygame.mask.Mask, msg)
            self.assertEqual(mask.get_size(), expected_size, msg)
            self.assertEqual(mask.count(), expected_count, msg)
            self.assertEqual(mask.overlap_area(expected_mask, offset), expected_count, msg)

    def test_from_surface__different_alphas_16bit(self):
        """Ensures from_surface creates a mask with the correct bits set
        when pixels have different alpha values (16 bit surfaces).

        This test checks the masks created by the from_surface function using
        a 16 bit surface. Each pixel of the surface is set with a different
        alpha value (0-255), but since this is a 16 bit surface the requested
        alpha value can differ from what is actually set. The resulting surface
        will have groups of alpha values which complicates the test as the
        alpha groups will all be set/unset at a given threshold. The setup
        calculates these groups and an expected mask for each. This test data
        is then used to test each alpha grouping over a range of threshold
        values.
        """
        threshold_count = 256
        surface_color = [110, 120, 130, 0]
        expected_size = (threshold_count, 1)
        surface = pygame.Surface(expected_size, SRCALPHA, 16)
        surface.lock()
        for a in range(threshold_count):
            surface_color[3] = a
            surface.set_at((a, 0), surface_color)
        surface.unlock()
        alpha_thresholds = OrderedDict()
        special_thresholds = set()
        for threshold in range(threshold_count):
            alpha = surface.get_at((threshold, 0))[3]
            if alpha not in alpha_thresholds:
                alpha_thresholds[alpha] = [threshold]
            else:
                alpha_thresholds[alpha].append(threshold)
            if threshold < alpha:
                special_thresholds.add(threshold)
        test_data = []
        offset = (0, 0)
        erase_mask = pygame.Mask(expected_size)
        exp_mask = pygame.Mask(expected_size, fill=True)
        for thresholds in alpha_thresholds.values():
            for threshold in thresholds:
                if threshold in special_thresholds:
                    test_data.append((threshold, threshold + 1, exp_mask))
                else:
                    to_threshold = thresholds[-1] + 1
                    for thres in range(to_threshold):
                        erase_mask.set_at((thres, 0), 1)
                    exp_mask = pygame.Mask(expected_size, fill=True)
                    exp_mask.erase(erase_mask, offset)
                    test_data.append((threshold, to_threshold, exp_mask))
                    break
        for from_threshold, to_threshold, expected_mask in test_data:
            expected_count = expected_mask.count()
            for threshold in range(from_threshold, to_threshold):
                msg = f'threshold={threshold}'
                mask = pygame.mask.from_surface(surface, threshold)
                self.assertIsInstance(mask, pygame.mask.Mask, msg)
                self.assertEqual(mask.get_size(), expected_size, msg)
                self.assertEqual(mask.count(), expected_count, msg)
                self.assertEqual(mask.overlap_area(expected_mask, offset), expected_count, msg)

    def test_from_surface__with_colorkey_mask_cleared(self):
        """Ensures from_surface creates a mask with the correct bits set
        when the surface uses a colorkey.

        The surface is filled with the colorkey color so the resulting masks
        are expected to have no bits set.
        """
        colorkeys = ((0, 0, 0), (1, 2, 3), (50, 100, 200), (255, 255, 255))
        expected_size = (7, 11)
        expected_count = 0
        for depth in (8, 16, 24, 32):
            msg = f'depth={depth}'
            surface = pygame.Surface(expected_size, 0, depth)
            for colorkey in colorkeys:
                surface.set_colorkey(colorkey)
                surface.fill(surface.get_colorkey())
                mask = pygame.mask.from_surface(surface)
                self.assertIsInstance(mask, pygame.mask.Mask, msg)
                self.assertEqual(mask.get_size(), expected_size, msg)
                self.assertEqual(mask.count(), expected_count, msg)

    def test_from_surface__with_colorkey_mask_filled(self):
        """Ensures from_surface creates a mask with the correct bits set
        when the surface uses a colorkey.

        The surface is filled with a color that is not the colorkey color so
        the resulting masks are expected to have all bits set.
        """
        colorkeys = ((0, 0, 0), (1, 2, 3), (10, 100, 200), (255, 255, 255))
        surface_color = (50, 100, 200)
        expected_size = (11, 7)
        expected_count = expected_size[0] * expected_size[1]
        for depth in (8, 16, 24, 32):
            msg = f'depth={depth}'
            surface = pygame.Surface(expected_size, 0, depth)
            surface.fill(surface_color)
            for colorkey in colorkeys:
                surface.set_colorkey(colorkey)
                mask = pygame.mask.from_surface(surface)
                self.assertIsInstance(mask, pygame.mask.Mask, msg)
                self.assertEqual(mask.get_size(), expected_size, msg)
                self.assertEqual(mask.count(), expected_count, msg)

    def test_from_surface__with_colorkey_mask_pattern(self):
        """Ensures from_surface creates a mask with the correct bits set
        when the surface uses a colorkey.

        The surface is filled with alternating pixels of colorkey and
        non-colorkey colors, so the resulting masks are expected to have
        alternating bits set.
        """

        def alternate(func, set_value, unset_value, width, height):
            setbit = False
            for pos in ((x, y) for x in range(width) for y in range(height)):
                func(pos, set_value if setbit else unset_value)
                setbit = not setbit
        surface_color = (5, 10, 20)
        colorkey = (50, 60, 70)
        expected_size = (11, 2)
        expected_mask = pygame.mask.Mask(expected_size)
        alternate(expected_mask.set_at, 1, 0, *expected_size)
        expected_count = expected_mask.count()
        offset = (0, 0)
        for depth in (8, 16, 24, 32):
            msg = f'depth={depth}'
            surface = pygame.Surface(expected_size, 0, depth)
            alternate(surface.set_at, surface_color, colorkey, *expected_size)
            surface.set_colorkey(colorkey)
            mask = pygame.mask.from_surface(surface)
            self.assertIsInstance(mask, pygame.mask.Mask, msg)
            self.assertEqual(mask.get_size(), expected_size, msg)
            self.assertEqual(mask.count(), expected_count, msg)
            self.assertEqual(mask.overlap_area(expected_mask, offset), expected_count, msg)

    def test_from_threshold(self):
        """Does mask.from_threshold() work correctly?"""
        a = [16, 24, 32]
        for i in a:
            surf = pygame.surface.Surface((70, 70), 0, i)
            surf.fill((100, 50, 200), (20, 20, 20, 20))
            mask = pygame.mask.from_threshold(surf, (100, 50, 200, 255), (10, 10, 10, 255))
            rects = mask.get_bounding_rects()
            self.assertEqual(mask.count(), 400)
            self.assertEqual(mask.get_bounding_rects(), [pygame.Rect((20, 20, 20, 20))])
        for i in a:
            surf = pygame.surface.Surface((70, 70), 0, i)
            surf2 = pygame.surface.Surface((70, 70), 0, i)
            surf.fill((100, 100, 100))
            surf2.fill((150, 150, 150))
            surf2.fill((100, 100, 100), (40, 40, 10, 10))
            mask = pygame.mask.from_threshold(surface=surf, color=(0, 0, 0, 0), threshold=(10, 10, 10, 255), othersurface=surf2)
            self.assertIsInstance(mask, pygame.mask.Mask)
            self.assertEqual(mask.count(), 100)
            self.assertEqual(mask.get_bounding_rects(), [pygame.Rect((40, 40, 10, 10))])

    def test_zero_size_from_surface(self):
        """Ensures from_surface can create masks from zero sized surfaces."""
        for size in ((100, 0), (0, 100), (0, 0)):
            mask = pygame.mask.from_surface(pygame.Surface(size))
            self.assertIsInstance(mask, pygame.mask.MaskType, f'size={size}')
            self.assertEqual(mask.get_size(), size)

    def test_zero_size_from_threshold(self):
        a = [16, 24, 32]
        sizes = ((100, 0), (0, 100), (0, 0))
        for size in sizes:
            for i in a:
                surf = pygame.surface.Surface(size, 0, i)
                surf.fill((100, 50, 200), (20, 20, 20, 20))
                mask = pygame.mask.from_threshold(surf, (100, 50, 200, 255), (10, 10, 10, 255))
                self.assertEqual(mask.count(), 0)
                rects = mask.get_bounding_rects()
                self.assertEqual(rects, [])
            for i in a:
                surf = pygame.surface.Surface(size, 0, i)
                surf2 = pygame.surface.Surface(size, 0, i)
                surf.fill((100, 100, 100))
                surf2.fill((150, 150, 150))
                surf2.fill((100, 100, 100), (40, 40, 10, 10))
                mask = pygame.mask.from_threshold(surf, (0, 0, 0, 0), (10, 10, 10, 255), surf2)
                self.assertIsInstance(mask, pygame.mask.Mask)
                self.assertEqual(mask.count(), 0)
                rects = mask.get_bounding_rects()
                self.assertEqual(rects, [])

    def test_buffer_interface(self):
        size = (1000, 100)
        pixels_set = ((0, 1), (100, 10), (173, 90))
        pixels_unset = ((0, 0), (101, 10), (173, 91))
        mask = pygame.Mask(size)
        for point in pixels_set:
            mask.set_at(point, 1)
        view = memoryview(mask)
        intwidth = 8 * view.strides[1]
        for point in pixels_set:
            x, y = point
            col = x // intwidth
            self.assertEqual(view[col, y] >> x % intwidth & 1, 1, f'the pixel at {point} is not set to 1')
        for point in pixels_unset:
            x, y = point
            col = x // intwidth
            self.assertEqual(view[col, y] >> x % intwidth & 1, 0, f'the pixel at {point} is not set to 0')