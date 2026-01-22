import gc
import operator
import platform
import sys
import unittest
import weakref
from functools import reduce
from pygame.tests.test_utils import SurfaceSubclass
import pygame
@unittest.skipIf(IS_PYPY, 'pypy having issues')
class PixelArrayArrayInterfaceTest(unittest.TestCase, TestMixin):

    @unittest.skipIf(IS_PYPY, 'skipping for PyPy (why?)')
    def test_basic(self):
        sf = pygame.Surface((2, 2), 0, 32)
        ar = pygame.PixelArray(sf)
        ai = arrinter.ArrayInterface(ar)
        self.assertEqual(ai.two, 2)
        self.assertEqual(ai.typekind, 'u')
        self.assertEqual(ai.nd, 2)
        self.assertEqual(ai.data, ar._pixels_address)

    @unittest.skipIf(IS_PYPY, 'skipping for PyPy (why?)')
    def test_shape(self):
        for shape in [[4, 16], [5, 13]]:
            w, h = shape
            sf = pygame.Surface(shape, 0, 32)
            ar = pygame.PixelArray(sf)
            ai = arrinter.ArrayInterface(ar)
            ai_shape = [ai.shape[i] for i in range(ai.nd)]
            self.assertEqual(ai_shape, shape)
            ar2 = ar[::2, :]
            ai2 = arrinter.ArrayInterface(ar2)
            w2 = len(([0] * w)[::2])
            ai_shape = [ai2.shape[i] for i in range(ai2.nd)]
            self.assertEqual(ai_shape, [w2, h])
            ar2 = ar[:, ::2]
            ai2 = arrinter.ArrayInterface(ar2)
            h2 = len(([0] * h)[::2])
            ai_shape = [ai2.shape[i] for i in range(ai2.nd)]
            self.assertEqual(ai_shape, [w, h2])

    @unittest.skipIf(IS_PYPY, 'skipping for PyPy (why?)')
    def test_itemsize(self):
        for bytes_per_pixel in range(1, 5):
            bits_per_pixel = 8 * bytes_per_pixel
            sf = pygame.Surface((2, 2), 0, bits_per_pixel)
            ar = pygame.PixelArray(sf)
            ai = arrinter.ArrayInterface(ar)
            self.assertEqual(ai.itemsize, bytes_per_pixel)

    @unittest.skipIf(IS_PYPY, 'skipping for PyPy (why?)')
    def test_flags(self):
        aim = arrinter
        common_flags = aim.PAI_NOTSWAPPED | aim.PAI_WRITEABLE | aim.PAI_ALIGNED
        s = pygame.Surface((10, 2), 0, 32)
        ar = pygame.PixelArray(s)
        ai = aim.ArrayInterface(ar)
        self.assertEqual(ai.flags, common_flags | aim.PAI_FORTRAN)
        ar2 = ar[::2, :]
        ai = aim.ArrayInterface(ar2)
        self.assertEqual(ai.flags, common_flags)
        s = pygame.Surface((8, 2), 0, 24)
        ar = pygame.PixelArray(s)
        ai = aim.ArrayInterface(ar)
        self.assertEqual(ai.flags, common_flags | aim.PAI_FORTRAN)
        s = pygame.Surface((7, 2), 0, 24)
        ar = pygame.PixelArray(s)
        ai = aim.ArrayInterface(ar)
        self.assertEqual(ai.flags, common_flags)

    def test_slicing(self):
        factors = [7, 3, 11]
        w = reduce(operator.mul, factors, 1)
        h = 13
        sf = pygame.Surface((w, h), 0, 8)
        color = sf.map_rgb((1, 17, 128))
        ar = pygame.PixelArray(sf)
        for f in factors[:-1]:
            w = w // f
            sf.fill((0, 0, 0))
            ar = ar[f:f + w, :]
            ar[0][0] = color
            ar[-1][-2] = color
            ar[0][-3] = color
            sf2 = ar.make_surface()
            sf3 = pygame.pixelcopy.make_surface(ar)
            self.assert_surfaces_equal(sf3, sf2)
        h = reduce(operator.mul, factors, 1)
        w = 13
        sf = pygame.Surface((w, h), 0, 8)
        color = sf.map_rgb((1, 17, 128))
        ar = pygame.PixelArray(sf)
        for f in factors[:-1]:
            h = h // f
            sf.fill((0, 0, 0))
            ar = ar[:, f:f + h]
            ar[0][0] = color
            ar[-1][-2] = color
            ar[0][-3] = color
            sf2 = ar.make_surface()
            sf3 = pygame.pixelcopy.make_surface(ar)
            self.assert_surfaces_equal(sf3, sf2)
        w = 20
        h = 10
        sf = pygame.Surface((w, h), 0, 8)
        color = sf.map_rgb((1, 17, 128))
        ar = pygame.PixelArray(sf)
        for slices in [(slice(w), slice(h)), (slice(0, w, 2), slice(h)), (slice(0, w, 3), slice(h)), (slice(w), slice(0, h, 2)), (slice(w), slice(0, h, 3)), (slice(0, w, 2), slice(0, h, 2)), (slice(0, w, 3), slice(0, h, 3))]:
            sf.fill((0, 0, 0))
            ar2 = ar[slices]
            ar2[0][0] = color
            ar2[-1][-2] = color
            ar2[0][-3] = color
            sf2 = ar2.make_surface()
            sf3 = pygame.pixelcopy.make_surface(ar2)
            self.assert_surfaces_equal(sf3, sf2)