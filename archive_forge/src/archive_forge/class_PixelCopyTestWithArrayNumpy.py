import platform
import unittest
import pygame
from pygame.locals import *
from pygame.pixelcopy import surface_to_array, map_array, array_to_surface, make_surface
@unittest.skipIf(IS_PYPY, 'pypy having illegal instruction on mac')
class PixelCopyTestWithArrayNumpy(unittest.TestCase):
    try:
        import numpy
    except ImportError:
        __tags__ = ['ignore', 'subprocess_ignore']
    else:
        pygame.surfarray.use_arraytype('numpy')
    bitsizes = [8, 16, 32]
    test_palette = [(0, 0, 0, 255), (10, 30, 60, 255), (25, 75, 100, 255), (100, 150, 200, 255), (0, 100, 200, 255)]
    surf_size = (10, 12)
    test_points = [((0, 0), 1), ((4, 5), 1), ((9, 0), 2), ((5, 5), 2), ((0, 11), 3), ((4, 6), 3), ((9, 11), 4), ((5, 6), 4)]
    pixels2d = {8, 16, 32}
    pixels3d = {24, 32}
    array2d = {8, 16, 24, 32}
    array3d = {24, 32}

    def __init__(self, *args, **kwds):
        import numpy
        self.dst_types = [numpy.uint8, numpy.uint16, numpy.uint32]
        try:
            self.dst_types.append(numpy.uint64)
        except AttributeError:
            pass
        pygame.display.init()
        try:
            unittest.TestCase.__init__(self, *args, **kwds)
            self.sources = [self._make_src_surface(8), self._make_src_surface(16), self._make_src_surface(16, srcalpha=True), self._make_src_surface(24), self._make_src_surface(32), self._make_src_surface(32, srcalpha=True)]
        finally:
            pygame.display.quit()

    def _make_surface(self, bitsize, srcalpha=False, palette=None):
        if palette is None:
            palette = self.test_palette
        flags = 0
        if srcalpha:
            flags |= SRCALPHA
        surf = pygame.Surface(self.surf_size, flags, bitsize)
        if bitsize == 8:
            surf.set_palette([c[:3] for c in palette])
        return surf

    def _fill_surface(self, surf, palette=None):
        if palette is None:
            palette = self.test_palette
        surf.fill(palette[1], (0, 0, 5, 6))
        surf.fill(palette[2], (5, 0, 5, 6))
        surf.fill(palette[3], (0, 6, 5, 6))
        surf.fill(palette[4], (5, 6, 5, 6))

    def _make_src_surface(self, bitsize, srcalpha=False, palette=None):
        surf = self._make_surface(bitsize, srcalpha, palette)
        self._fill_surface(surf, palette)
        return surf

    def setUp(self):
        pygame.display.init()

    def tearDown(self):
        pygame.display.quit()

    def test_surface_to_array_2d(self):
        try:
            from numpy import empty, dtype
        except ImportError:
            return
        palette = self.test_palette
        alpha_color = (0, 0, 0, 128)
        dst_dims = self.surf_size
        destinations = [empty(dst_dims, t) for t in self.dst_types]
        if pygame.get_sdl_byteorder() == pygame.LIL_ENDIAN:
            swapped_dst = empty(dst_dims, dtype('>u4'))
        else:
            swapped_dst = empty(dst_dims, dtype('<u4'))
        for surf in self.sources:
            src_bytesize = surf.get_bytesize()
            for dst in destinations:
                if dst.itemsize < src_bytesize:
                    self.assertRaises(ValueError, surface_to_array, dst, surf)
                    continue
                dst[...] = 0
                self.assertFalse(surf.get_locked())
                surface_to_array(dst, surf)
                self.assertFalse(surf.get_locked())
                for posn, i in self.test_points:
                    sp = unsigned32(surf.get_at_mapped(posn))
                    dp = dst[posn]
                    self.assertEqual(dp, sp, '%s != %s: flags: %i, bpp: %i, dtype: %s,  posn: %s' % (dp, sp, surf.get_flags(), surf.get_bitsize(), dst.dtype, posn))
                if surf.get_masks()[3]:
                    posn = (2, 1)
                    surf.set_at(posn, alpha_color)
                    surface_to_array(dst, surf)
                    sp = unsigned32(surf.get_at_mapped(posn))
                    dp = dst[posn]
                    self.assertEqual(dp, sp, '%s != %s: bpp: %i' % (dp, sp, surf.get_bitsize()))
            swapped_dst[...] = 0
            self.assertFalse(surf.get_locked())
            surface_to_array(swapped_dst, surf)
            self.assertFalse(surf.get_locked())
            for posn, i in self.test_points:
                sp = unsigned32(surf.get_at_mapped(posn))
                dp = swapped_dst[posn]
                self.assertEqual(dp, sp, '%s != %s: flags: %i, bpp: %i, dtype: %s,  posn: %s' % (dp, sp, surf.get_flags(), surf.get_bitsize(), dst.dtype, posn))
            if surf.get_masks()[3]:
                posn = (2, 1)
                surf.set_at(posn, alpha_color)
                self.assertFalse(surf.get_locked())
                surface_to_array(swapped_dst, surf)
                self.assertFalse(surf.get_locked())
                sp = unsigned32(surf.get_at_mapped(posn))
                dp = swapped_dst[posn]
                self.assertEqual(dp, sp, '%s != %s: bpp: %i' % (dp, sp, surf.get_bitsize()))

    def test_surface_to_array_3d(self):
        try:
            from numpy import empty, dtype
        except ImportError:
            return
        palette = self.test_palette
        dst_dims = self.surf_size + (3,)
        destinations = [empty(dst_dims, t) for t in self.dst_types]
        if pygame.get_sdl_byteorder() == pygame.LIL_ENDIAN:
            swapped_dst = empty(dst_dims, dtype('>u4'))
        else:
            swapped_dst = empty(dst_dims, dtype('<u4'))
        for surf in self.sources:
            src_bitsize = surf.get_bitsize()
            for dst in destinations:
                dst[...] = 0
                self.assertFalse(surf.get_locked())
                surface_to_array(dst, surf)
                self.assertFalse(surf.get_locked())
                for posn, i in self.test_points:
                    r_surf, g_surf, b_surf, a_surf = surf.get_at(posn)
                    r_arr, g_arr, b_arr = dst[posn]
                    self.assertEqual(r_arr, r_surf, '%i != %i, color: red, flags: %i, bpp: %i, posn: %s' % (r_arr, r_surf, surf.get_flags(), surf.get_bitsize(), posn))
                    self.assertEqual(g_arr, g_surf, '%i != %i, color: green, flags: %i, bpp: %i, posn: %s' % (r_arr, r_surf, surf.get_flags(), surf.get_bitsize(), posn))
                    self.assertEqual(b_arr, b_surf, '%i != %i, color: blue, flags: %i, bpp: %i, posn: %s' % (r_arr, r_surf, surf.get_flags(), surf.get_bitsize(), posn))
            swapped_dst[...] = 0
            self.assertFalse(surf.get_locked())
            surface_to_array(swapped_dst, surf)
            self.assertFalse(surf.get_locked())
            for posn, i in self.test_points:
                r_surf, g_surf, b_surf, a_surf = surf.get_at(posn)
                r_arr, g_arr, b_arr = swapped_dst[posn]
                self.assertEqual(r_arr, r_surf, '%i != %i, color: red, flags: %i, bpp: %i, posn: %s' % (r_arr, r_surf, surf.get_flags(), surf.get_bitsize(), posn))
                self.assertEqual(g_arr, g_surf, '%i != %i, color: green, flags: %i, bpp: %i, posn: %s' % (r_arr, r_surf, surf.get_flags(), surf.get_bitsize(), posn))
                self.assertEqual(b_arr, b_surf, '%i != %i, color: blue, flags: %i, bpp: %i, posn: %s' % (r_arr, r_surf, surf.get_flags(), surf.get_bitsize(), posn))

    def test_map_array(self):
        try:
            from numpy import array, zeros, uint8, int32, alltrue
        except ImportError:
            return
        surf = pygame.Surface((1, 1), 0, 32)
        color = array([11, 17, 59], uint8)
        target = zeros((5, 7), int32)
        map_array(target, color, surf)
        self.assertTrue(alltrue(target == surf.map_rgb(color)))
        stripe = array([[2, 5, 7], [11, 19, 23], [37, 53, 101]], uint8)
        target = zeros((4, stripe.shape[0]), int32)
        map_array(target, stripe, surf)
        target_stripe = array([surf.map_rgb(c) for c in stripe], int32)
        self.assertTrue(alltrue(target == target_stripe))
        stripe = array([[[2, 5, 7]], [[11, 19, 24]], [[10, 20, 30]], [[37, 53, 101]]], uint8)
        target = zeros((stripe.shape[0], 3), int32)
        map_array(target, stripe, surf)
        target_stripe = array([[surf.map_rgb(c)] for c in stripe[:, 0]], int32)
        self.assertTrue(alltrue(target == target_stripe))
        w = 4
        h = 5
        source = zeros((w, h, 3), uint8)
        target = zeros((w,), int32)
        self.assertRaises(ValueError, map_array, target, source, surf)
        source = zeros((12, w, h + 1), uint8)
        self.assertRaises(ValueError, map_array, target, source, surf)
        source = zeros((12, w - 1, 5), uint8)
        self.assertRaises(ValueError, map_array, target, source, surf)
    try:
        numpy
    except NameError:
        del __init__
        del test_surface_to_array_2d
        del test_surface_to_array_3d
        del test_map_array
    else:
        del numpy