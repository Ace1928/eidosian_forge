import os
import unittest
from pygame.tests import test_utils
from pygame.tests.test_utils import (
import pygame
from pygame.locals import *
from pygame.bufferproxy import BufferProxy
import platform
import gc
import weakref
import ctypes
class SurfaceTypeTest(unittest.TestCase):

    def test_surface__pixel_format_as_surface_subclass(self):
        """Ensure a subclassed surface can be used for pixel format
        when creating a new surface."""
        expected_depth = 16
        expected_flags = SRCALPHA
        expected_size = (13, 37)
        depth_surface = SurfaceSubclass((11, 21), expected_flags, expected_depth)
        surface = pygame.Surface(expected_size, expected_flags, depth_surface)
        self.assertIsNot(surface, depth_surface)
        self.assertIsInstance(surface, pygame.Surface)
        self.assertNotIsInstance(surface, SurfaceSubclass)
        self.assertEqual(surface.get_size(), expected_size)
        self.assertEqual(surface.get_flags(), expected_flags)
        self.assertEqual(surface.get_bitsize(), expected_depth)

    def test_surface_created_opaque_black(self):
        surf = pygame.Surface((20, 20))
        self.assertEqual(surf.get_at((0, 0)), (0, 0, 0, 255))
        pygame.display.set_mode((500, 500))
        surf = pygame.Surface((20, 20))
        self.assertEqual(surf.get_at((0, 0)), (0, 0, 0, 255))

    def test_set_clip(self):
        """see if surface.set_clip(None) works correctly."""
        s = pygame.Surface((800, 600))
        r = pygame.Rect(10, 10, 10, 10)
        s.set_clip(r)
        r.move_ip(10, 0)
        s.set_clip(None)
        res = s.get_clip()
        self.assertEqual(res[0], 0)
        self.assertEqual(res[2], 800)

    def test_print(self):
        surf = pygame.Surface((70, 70), 0, 32)
        self.assertEqual(repr(surf), '<Surface(70x70x32 SW)>')

    def test_keyword_arguments(self):
        surf = pygame.Surface((70, 70), flags=SRCALPHA, depth=32)
        self.assertEqual(surf.get_flags() & SRCALPHA, SRCALPHA)
        self.assertEqual(surf.get_bitsize(), 32)
        surf_16 = pygame.Surface((70, 70), 0, 16)
        self.assertEqual(surf_16.get_bytesize(), 2)
        surf_16 = pygame.Surface((70, 70), depth=16)
        self.assertEqual(surf_16.get_bytesize(), 2)

    def test_set_at(self):
        s = pygame.Surface((100, 100), 0, 24)
        s.fill((0, 0, 0))
        s.set_at((0, 0), (10, 10, 10, 255))
        r = s.get_at((0, 0))
        self.assertIsInstance(r, pygame.Color)
        self.assertEqual(r, (10, 10, 10, 255))
        s.fill((0, 0, 0, 255))
        s.set_at((10, 1), 255)
        r = s.get_at((10, 1))
        self.assertEqual(r, (0, 0, 255, 255))

    def test_set_at__big_endian(self):
        """png files are loaded in big endian format (BGR rather than RGB)"""
        pygame.display.init()
        try:
            image = pygame.image.load(example_path(os.path.join('data', 'BGR.png')))
            self.assertEqual(image.get_at((10, 10)), pygame.Color(255, 0, 0))
            self.assertEqual(image.get_at((10, 20)), pygame.Color(0, 255, 0))
            self.assertEqual(image.get_at((10, 40)), pygame.Color(0, 0, 255))
            image.set_at((10, 10), pygame.Color(255, 0, 0))
            image.set_at((10, 20), pygame.Color(0, 255, 0))
            image.set_at((10, 40), pygame.Color(0, 0, 255))
            self.assertEqual(image.get_at((10, 10)), pygame.Color(255, 0, 0))
            self.assertEqual(image.get_at((10, 20)), pygame.Color(0, 255, 0))
            self.assertEqual(image.get_at((10, 40)), pygame.Color(0, 0, 255))
        finally:
            pygame.display.quit()

    def test_SRCALPHA(self):
        surf = pygame.Surface((70, 70), SRCALPHA, 32)
        self.assertEqual(surf.get_flags() & SRCALPHA, SRCALPHA)
        self.assertRaises(ValueError, pygame.Surface, (100, 100), pygame.SRCALPHA, 24)
        surf2 = pygame.Surface((70, 70), SRCALPHA)
        if surf2.get_bitsize() == 32:
            self.assertEqual(surf2.get_flags() & SRCALPHA, SRCALPHA)

    def test_flags_default0_nodisplay(self):
        """is set to zero, and SRCALPHA is not set by default with no display initialized."""
        pygame.display.quit()
        surf = pygame.Surface((70, 70))
        self.assertEqual(surf.get_flags() & SRCALPHA, 0)

    def test_flags_default0_display(self):
        """is set to zero, and SRCALPH is not set by default even when the display is initialized."""
        pygame.display.set_mode((320, 200))
        try:
            surf = pygame.Surface((70, 70))
            self.assertEqual(surf.get_flags() & SRCALPHA, 0)
        finally:
            pygame.display.quit()

    def test_masks(self):

        def make_surf(bpp, flags, masks):
            pygame.Surface((10, 10), flags, bpp, masks)
        masks = (4278190080, 16711680, 65280, 0)
        self.assertEqual(make_surf(32, 0, masks), None)
        masks = (8323072, 65280, 255, 0)
        self.assertRaises(ValueError, make_surf, 24, 0, masks)
        self.assertRaises(ValueError, make_surf, 32, 0, masks)
        masks = (7274496, 65280, 255, 0)
        self.assertRaises(ValueError, make_surf, 32, 0, masks)

    def test_get_bounding_rect(self):
        surf = pygame.Surface((70, 70), SRCALPHA, 32)
        surf.fill((0, 0, 0, 0))
        bound_rect = surf.get_bounding_rect()
        self.assertEqual(bound_rect.width, 0)
        self.assertEqual(bound_rect.height, 0)
        surf.set_at((30, 30), (255, 255, 255, 1))
        bound_rect = surf.get_bounding_rect()
        self.assertEqual(bound_rect.left, 30)
        self.assertEqual(bound_rect.top, 30)
        self.assertEqual(bound_rect.width, 1)
        self.assertEqual(bound_rect.height, 1)
        surf.set_at((29, 29), (255, 255, 255, 1))
        bound_rect = surf.get_bounding_rect()
        self.assertEqual(bound_rect.left, 29)
        self.assertEqual(bound_rect.top, 29)
        self.assertEqual(bound_rect.width, 2)
        self.assertEqual(bound_rect.height, 2)
        surf = pygame.Surface((70, 70), 0, 24)
        surf.fill((0, 0, 0))
        bound_rect = surf.get_bounding_rect()
        self.assertEqual(bound_rect.width, surf.get_width())
        self.assertEqual(bound_rect.height, surf.get_height())
        surf.set_colorkey((0, 0, 0))
        bound_rect = surf.get_bounding_rect()
        self.assertEqual(bound_rect.width, 0)
        self.assertEqual(bound_rect.height, 0)
        surf.set_at((30, 30), (255, 255, 255))
        bound_rect = surf.get_bounding_rect()
        self.assertEqual(bound_rect.left, 30)
        self.assertEqual(bound_rect.top, 30)
        self.assertEqual(bound_rect.width, 1)
        self.assertEqual(bound_rect.height, 1)
        surf.set_at((60, 60), (255, 255, 255))
        bound_rect = surf.get_bounding_rect()
        self.assertEqual(bound_rect.left, 30)
        self.assertEqual(bound_rect.top, 30)
        self.assertEqual(bound_rect.width, 31)
        self.assertEqual(bound_rect.height, 31)
        pygame.display.init()
        try:
            surf = pygame.Surface((4, 1), 0, 8)
            surf.fill((255, 255, 255))
            surf.get_bounding_rect()
        finally:
            pygame.display.quit()

    def test_copy(self):
        """Ensure a surface can be copied."""
        color = (25, 25, 25, 25)
        s1 = pygame.Surface((32, 32), pygame.SRCALPHA, 32)
        s1.fill(color)
        s2 = s1.copy()
        s1rect = s1.get_rect()
        s2rect = s2.get_rect()
        self.assertEqual(s1rect.size, s2rect.size)
        self.assertEqual(s2.get_at((10, 10)), color)

    def test_fill(self):
        """Ensure a surface can be filled."""
        color = (25, 25, 25, 25)
        fill_rect = pygame.Rect(0, 0, 16, 16)
        s1 = pygame.Surface((32, 32), pygame.SRCALPHA, 32)
        s1.fill(color, fill_rect)
        for pt in test_utils.rect_area_pts(fill_rect):
            self.assertEqual(s1.get_at(pt), color)
        for pt in test_utils.rect_outer_bounds(fill_rect):
            self.assertNotEqual(s1.get_at(pt), color)

    def test_fill_rle(self):
        """Test RLEACCEL flag with fill()"""
        color = (250, 25, 25, 255)
        surf = pygame.Surface((32, 32))
        blit_surf = pygame.Surface((32, 32))
        blit_surf.set_colorkey((255, 0, 255), pygame.RLEACCEL)
        self.assertTrue(blit_surf.get_flags() & pygame.RLEACCELOK)
        surf.blit(blit_surf, (0, 0))
        blit_surf.fill(color)
        self.assertEqual(blit_surf.mustlock(), blit_surf.get_flags() & pygame.RLEACCEL != 0)
        self.assertTrue(blit_surf.get_flags() & pygame.RLEACCEL)

    def test_mustlock_rle(self):
        """Test RLEACCEL flag with mustlock()"""
        surf = pygame.Surface((100, 100))
        blit_surf = pygame.Surface((100, 100))
        blit_surf.set_colorkey((0, 0, 255), pygame.RLEACCEL)
        self.assertTrue(blit_surf.get_flags() & pygame.RLEACCELOK)
        surf.blit(blit_surf, (0, 0))
        self.assertTrue(blit_surf.get_flags() & pygame.RLEACCEL)
        self.assertTrue(blit_surf.mustlock())

    def test_mustlock_surf_alpha_rle(self):
        """Test RLEACCEL flag with mustlock() on a surface
        with per pixel alpha - new feature in SDL2"""
        surf = pygame.Surface((100, 100))
        blit_surf = pygame.Surface((100, 100), depth=32, flags=pygame.SRCALPHA)
        blit_surf.set_colorkey((192, 191, 192, 255), pygame.RLEACCEL)
        self.assertTrue(blit_surf.get_flags() & pygame.RLEACCELOK)
        surf.blit(blit_surf, (0, 0))
        self.assertTrue(blit_surf.get_flags() & pygame.RLEACCEL)
        self.assertTrue(blit_surf.get_flags() & pygame.SRCALPHA)
        self.assertTrue(blit_surf.mustlock())

    def test_copy_rle(self):
        """Test copying a surface set to use run length encoding"""
        s1 = pygame.Surface((32, 32), 24)
        s1.set_colorkey((255, 0, 255), pygame.RLEACCEL)
        self.assertTrue(s1.get_flags() & pygame.RLEACCELOK)
        newsurf = s1.copy()
        self.assertTrue(s1.get_flags() & pygame.RLEACCELOK)
        self.assertTrue(newsurf.get_flags() & pygame.RLEACCELOK)

    def test_subsurface_rle(self):
        """Ensure an RLE sub-surface works independently of its parent."""
        color = (250, 25, 25, 255)
        color2 = (200, 200, 250, 255)
        sub_rect = pygame.Rect(16, 16, 16, 16)
        s0 = pygame.Surface((32, 32), 24)
        s1 = pygame.Surface((32, 32), 24)
        s1.set_colorkey((255, 0, 255), pygame.RLEACCEL)
        s1.fill(color)
        s2 = s1.subsurface(sub_rect)
        s2.fill(color2)
        s0.blit(s1, (0, 0))
        self.assertTrue(s1.get_flags() & pygame.RLEACCEL)
        self.assertTrue(not s2.get_flags() & pygame.RLEACCEL)

    def test_subsurface_rle2(self):
        """Ensure an RLE sub-surface works independently of its parent."""
        color = (250, 25, 25, 255)
        color2 = (200, 200, 250, 255)
        sub_rect = pygame.Rect(16, 16, 16, 16)
        s0 = pygame.Surface((32, 32), 24)
        s1 = pygame.Surface((32, 32), 24)
        s1.set_colorkey((255, 0, 255), pygame.RLEACCEL)
        s1.fill(color)
        s2 = s1.subsurface(sub_rect)
        s2.fill(color2)
        s0.blit(s2, (0, 0))
        self.assertTrue(s1.get_flags() & pygame.RLEACCELOK)
        self.assertTrue(not s2.get_flags() & pygame.RLEACCELOK)

    def test_solarwolf_rle_usage(self):
        """Test for error/crash when calling set_colorkey() followed
        by convert twice in succession. Code originally taken
        from solarwolf."""

        def optimize(img):
            clear = img.get_colorkey()
            img.set_colorkey(clear, RLEACCEL)
            self.assertEqual(img.get_colorkey(), clear)
            return img.convert()
        pygame.display.init()
        try:
            pygame.display.set_mode((640, 480))
            image = pygame.image.load(example_path(os.path.join('data', 'alien1.png')))
            image = image.convert()
            orig_colorkey = image.get_colorkey()
            image = optimize(image)
            image = optimize(image)
            self.assertTrue(image.get_flags() & pygame.RLEACCELOK)
            self.assertTrue(not image.get_flags() & pygame.RLEACCEL)
            self.assertEqual(image.get_colorkey(), orig_colorkey)
            self.assertTrue(isinstance(image, pygame.Surface))
        finally:
            pygame.display.quit()

    def test_solarwolf_rle_usage_2(self):
        """Test for RLE status after setting alpha"""
        pygame.display.init()
        try:
            pygame.display.set_mode((640, 480), depth=32)
            blit_to_surf = pygame.Surface((100, 100))
            image = pygame.image.load(example_path(os.path.join('data', 'alien1.png')))
            image = image.convert()
            orig_colorkey = image.get_colorkey()
            image.set_colorkey(orig_colorkey, RLEACCEL)
            self.assertTrue(image.get_flags() & pygame.RLEACCELOK)
            self.assertTrue(not image.get_flags() & pygame.RLEACCEL)
            blit_to_surf.blit(image, (0, 0))
            self.assertTrue(image.get_flags() & pygame.RLEACCELOK)
            self.assertTrue(image.get_flags() & pygame.RLEACCEL)
            image.set_alpha(90)
            self.assertTrue(not image.get_flags() & pygame.RLEACCELOK)
            self.assertTrue(not image.get_flags() & pygame.RLEACCEL)
        finally:
            pygame.display.quit()

    def test_set_alpha__set_colorkey_rle(self):
        pygame.display.init()
        try:
            pygame.display.set_mode((640, 480))
            blit_to_surf = pygame.Surface((80, 71))
            blit_to_surf.fill((255, 255, 255))
            image = pygame.image.load(example_path(os.path.join('data', 'alien1.png')))
            image = image.convert()
            orig_colorkey = image.get_colorkey()
            image.set_alpha(90, RLEACCEL)
            blit_to_surf.blit(image, (0, 0))
            sample_pixel_rle = blit_to_surf.get_at((50, 50))
            self.assertEqual(image.get_colorkey(), orig_colorkey)
            image.set_colorkey(orig_colorkey, RLEACCEL)
            blit_to_surf.fill((255, 255, 255))
            blit_to_surf.blit(image, (0, 0))
            sample_pixel_no_rle = blit_to_surf.get_at((50, 50))
            self.assertAlmostEqual(sample_pixel_rle.r, sample_pixel_no_rle.r, delta=2)
            self.assertAlmostEqual(sample_pixel_rle.g, sample_pixel_no_rle.g, delta=2)
            self.assertAlmostEqual(sample_pixel_rle.b, sample_pixel_no_rle.b, delta=2)
        finally:
            pygame.display.quit()

    def test_fill_negative_coordinates(self):
        color = (25, 25, 25, 25)
        color2 = (20, 20, 20, 25)
        fill_rect = pygame.Rect(-10, -10, 16, 16)
        s1 = pygame.Surface((32, 32), pygame.SRCALPHA, 32)
        r1 = s1.fill(color, fill_rect)
        c = s1.get_at((0, 0))
        self.assertEqual(c, color)
        s2 = s1.subsurface((5, 5, 5, 5))
        r2 = s2.fill(color2, (-3, -3, 5, 5))
        c2 = s1.get_at((4, 4))
        self.assertEqual(c, color)
        r3 = s2.fill(color2, (-30, -30, 5, 5))
        self.assertEqual(tuple(r3), (0, 0, 0, 0))

    def test_fill_keyword_args(self):
        """Ensure fill() accepts keyword arguments."""
        color = (1, 2, 3, 255)
        area = (1, 1, 2, 2)
        s1 = pygame.Surface((4, 4), 0, 32)
        s1.fill(special_flags=pygame.BLEND_ADD, color=color, rect=area)
        self.assertEqual(s1.get_at((0, 0)), (0, 0, 0, 255))
        self.assertEqual(s1.get_at((1, 1)), color)

    def test_get_alpha(self):
        """Ensure a surface's alpha value can be retrieved."""
        s1 = pygame.Surface((32, 32), pygame.SRCALPHA, 32)
        self.assertEqual(s1.get_alpha(), 255)
        for alpha in (0, 32, 127, 255):
            s1.set_alpha(alpha)
            for t in range(4):
                s1.set_alpha(s1.get_alpha())
            self.assertEqual(s1.get_alpha(), alpha)

    def test_get_bytesize(self):
        """Ensure a surface's bit and byte sizes can be retrieved."""
        pygame.display.init()
        try:
            depth = 32
            depth_bytes = 4
            s1 = pygame.Surface((32, 32), pygame.SRCALPHA, depth)
            self.assertEqual(s1.get_bytesize(), depth_bytes)
            self.assertEqual(s1.get_bitsize(), depth)
            depth = 15
            depth_bytes = 2
            s1 = pygame.Surface((32, 32), 0, depth)
            self.assertEqual(s1.get_bytesize(), depth_bytes)
            self.assertEqual(s1.get_bitsize(), depth)
            depth = 12
            depth_bytes = 2
            s1 = pygame.Surface((32, 32), 0, depth)
            self.assertEqual(s1.get_bytesize(), depth_bytes)
            self.assertEqual(s1.get_bitsize(), depth)
            with self.assertRaises(pygame.error):
                surface = pygame.display.set_mode()
                pygame.display.quit()
                surface.get_bytesize()
        finally:
            pygame.display.quit()

    def test_get_flags(self):
        """Ensure a surface's flags can be retrieved."""
        s1 = pygame.Surface((32, 32), pygame.SRCALPHA, 32)
        self.assertEqual(s1.get_flags(), pygame.SRCALPHA)

    @unittest.skipIf(os.environ.get('SDL_VIDEODRIVER') == 'dummy', 'requires a non-"dummy" SDL_VIDEODRIVER')
    def test_get_flags__display_surf(self):
        pygame.display.init()
        try:
            screen_surf = pygame.display.set_mode((600, 400), flags=0)
            self.assertFalse(screen_surf.get_flags() & pygame.FULLSCREEN)
            screen_surf = pygame.display.set_mode((600, 400), flags=pygame.FULLSCREEN)
            self.assertTrue(screen_surf.get_flags() & pygame.FULLSCREEN)
            screen_surf = pygame.display.set_mode((600, 400), flags=0)
            self.assertFalse(screen_surf.get_flags() & pygame.NOFRAME)
            screen_surf = pygame.display.set_mode((600, 400), flags=pygame.NOFRAME)
            self.assertTrue(screen_surf.get_flags() & pygame.NOFRAME)
            screen_surf = pygame.display.set_mode((600, 400), flags=0)
            self.assertFalse(screen_surf.get_flags() & pygame.RESIZABLE)
            screen_surf = pygame.display.set_mode((600, 400), flags=pygame.RESIZABLE)
            self.assertTrue(screen_surf.get_flags() & pygame.RESIZABLE)
            screen_surf = pygame.display.set_mode((600, 400), flags=0)
            if not screen_surf.get_flags() & pygame.OPENGL:
                self.assertFalse(screen_surf.get_flags() & pygame.OPENGL)
            try:
                pygame.display.set_mode((200, 200), pygame.OPENGL, 32)
            except pygame.error:
                pass
            else:
                self.assertTrue(screen_surf.get_flags() & pygame.OPENGL)
        finally:
            pygame.display.quit()

    def test_get_parent(self):
        """Ensure a surface's parent can be retrieved."""
        pygame.display.init()
        try:
            parent = pygame.Surface((16, 16))
            child = parent.subsurface((0, 0, 5, 5))
            self.assertIs(child.get_parent(), parent)
            with self.assertRaises(pygame.error):
                surface = pygame.display.set_mode()
                pygame.display.quit()
                surface.get_parent()
        finally:
            pygame.display.quit()

    def test_get_rect(self):
        """Ensure a surface's rect can be retrieved."""
        size = (16, 16)
        surf = pygame.Surface(size)
        rect = surf.get_rect()
        self.assertEqual(rect.size, size)

    def test_get_width__size_and_height(self):
        """Ensure a surface's size, width and height can be retrieved."""
        for w in range(0, 255, 32):
            for h in range(0, 127, 15):
                s = pygame.Surface((w, h))
                self.assertEqual(s.get_width(), w)
                self.assertEqual(s.get_height(), h)
                self.assertEqual(s.get_size(), (w, h))

    def test_get_view(self):
        """Ensure a buffer view of the surface's pixels can be retrieved."""
        Error = ValueError
        s = pygame.Surface((5, 7), 0, 8)
        v2 = s.get_view('2')
        self.assertRaises(Error, s.get_view, '0')
        self.assertRaises(Error, s.get_view, '1')
        self.assertIsInstance(v2, BufferProxy)
        self.assertRaises(Error, s.get_view, '3')
        s = pygame.Surface((8, 7), 0, 8)
        length = s.get_bytesize() * s.get_width() * s.get_height()
        v0 = s.get_view('0')
        v1 = s.get_view('1')
        self.assertIsInstance(v0, BufferProxy)
        self.assertEqual(v0.length, length)
        self.assertIsInstance(v1, BufferProxy)
        self.assertEqual(v1.length, length)
        s = pygame.Surface((5, 7), 0, 16)
        v2 = s.get_view('2')
        self.assertRaises(Error, s.get_view, '0')
        self.assertRaises(Error, s.get_view, '1')
        self.assertIsInstance(v2, BufferProxy)
        self.assertRaises(Error, s.get_view, '3')
        s = pygame.Surface((8, 7), 0, 16)
        length = s.get_bytesize() * s.get_width() * s.get_height()
        v0 = s.get_view('0')
        v1 = s.get_view('1')
        self.assertIsInstance(v0, BufferProxy)
        self.assertEqual(v0.length, length)
        self.assertIsInstance(v1, BufferProxy)
        self.assertEqual(v1.length, length)
        s = pygame.Surface((5, 7), pygame.SRCALPHA, 16)
        v2 = s.get_view('2')
        self.assertIsInstance(v2, BufferProxy)
        self.assertRaises(Error, s.get_view, '3')
        s = pygame.Surface((5, 7), 0, 24)
        v2 = s.get_view('2')
        v3 = s.get_view('3')
        self.assertRaises(Error, s.get_view, '0')
        self.assertRaises(Error, s.get_view, '1')
        self.assertIsInstance(v2, BufferProxy)
        self.assertIsInstance(v3, BufferProxy)
        s = pygame.Surface((8, 7), 0, 24)
        length = s.get_bytesize() * s.get_width() * s.get_height()
        v0 = s.get_view('0')
        v1 = s.get_view('1')
        self.assertIsInstance(v0, BufferProxy)
        self.assertEqual(v0.length, length)
        self.assertIsInstance(v1, BufferProxy)
        self.assertEqual(v1.length, length)
        s = pygame.Surface((5, 7), 0, 32)
        length = s.get_bytesize() * s.get_width() * s.get_height()
        v0 = s.get_view('0')
        v1 = s.get_view('1')
        v2 = s.get_view('2')
        v3 = s.get_view('3')
        self.assertIsInstance(v0, BufferProxy)
        self.assertEqual(v0.length, length)
        self.assertIsInstance(v1, BufferProxy)
        self.assertEqual(v1.length, length)
        self.assertIsInstance(v2, BufferProxy)
        self.assertIsInstance(v3, BufferProxy)
        s2 = s.subsurface((0, 0, 4, 7))
        self.assertRaises(Error, s2.get_view, '0')
        self.assertRaises(Error, s2.get_view, '1')
        s2 = None
        s = pygame.Surface((5, 7), pygame.SRCALPHA, 32)
        for kind in ('2', '3', 'a', 'A', 'r', 'R', 'g', 'G', 'b', 'B'):
            self.assertIsInstance(s.get_view(kind), BufferProxy)
        s = pygame.Surface((2, 4), 0, 32)
        v = s.get_view()
        if not IS_PYPY:
            ai = ArrayInterface(v)
            self.assertEqual(ai.nd, 2)
        s = pygame.Surface((2, 4), 0, 32)
        self.assertFalse(s.get_locked())
        v = s.get_view('2')
        self.assertFalse(s.get_locked())
        c = v.__array_interface__
        self.assertTrue(s.get_locked())
        c = None
        gc.collect()
        self.assertTrue(s.get_locked())
        v = None
        gc.collect()
        self.assertFalse(s.get_locked())
        s = pygame.Surface((2, 4), pygame.SRCALPHA, 32)
        self.assertRaises(TypeError, s.get_view, '')
        self.assertRaises(TypeError, s.get_view, '9')
        self.assertRaises(TypeError, s.get_view, 'RGBA')
        self.assertRaises(TypeError, s.get_view, 2)
        s = pygame.Surface((2, 4), 0, 32)
        s.get_view('2')
        s.get_view(b'2')
        s = pygame.Surface((2, 4), 0, 32)
        weak_s = weakref.ref(s)
        v = s.get_view('3')
        weak_v = weakref.ref(v)
        gc.collect()
        self.assertTrue(weak_s() is s)
        self.assertTrue(weak_v() is v)
        del v
        gc.collect()
        self.assertTrue(weak_s() is s)
        self.assertTrue(weak_v() is None)
        del s
        gc.collect()
        self.assertTrue(weak_s() is None)

    def test_get_buffer(self):
        for bitsize in [8, 16, 24, 32]:
            s = pygame.Surface((5, 7), 0, bitsize)
            length = s.get_pitch() * s.get_height()
            v = s.get_buffer()
            self.assertIsInstance(v, BufferProxy)
            self.assertEqual(v.length, length)
            self.assertEqual(repr(v), f'<BufferProxy({length})>')
        s = pygame.Surface((7, 10), 0, 32)
        s2 = s.subsurface((1, 2, 5, 7))
        length = s2.get_pitch() * s2.get_height()
        v = s2.get_buffer()
        self.assertIsInstance(v, BufferProxy)
        self.assertEqual(v.length, length)
        s = pygame.Surface((2, 4), 0, 32)
        v = s.get_buffer()
        self.assertTrue(s.get_locked())
        v = None
        gc.collect()
        self.assertFalse(s.get_locked())
    OLDBUF = hasattr(pygame.bufferproxy, 'get_segcount')

    @unittest.skipIf(not OLDBUF, 'old buffer not available')
    def test_get_buffer_oldbuf(self):
        from pygame.bufferproxy import get_segcount, get_write_buffer
        s = pygame.Surface((2, 4), pygame.SRCALPHA, 32)
        v = s.get_buffer()
        segcount, buflen = get_segcount(v)
        self.assertEqual(segcount, 1)
        self.assertEqual(buflen, s.get_pitch() * s.get_height())
        seglen, segaddr = get_write_buffer(v, 0)
        self.assertEqual(segaddr, s._pixels_address)
        self.assertEqual(seglen, buflen)

    @unittest.skipIf(not OLDBUF, 'old buffer not available')
    def test_get_view_oldbuf(self):
        from pygame.bufferproxy import get_segcount, get_write_buffer
        s = pygame.Surface((2, 4), pygame.SRCALPHA, 32)
        v = s.get_view('1')
        segcount, buflen = get_segcount(v)
        self.assertEqual(segcount, 8)
        self.assertEqual(buflen, s.get_pitch() * s.get_height())
        seglen, segaddr = get_write_buffer(v, 7)
        self.assertEqual(segaddr, s._pixels_address + s.get_bytesize() * 7)
        self.assertEqual(seglen, s.get_bytesize())

    def test_set_colorkey(self):
        s = pygame.Surface((16, 16), pygame.SRCALPHA, 32)
        colorkeys = ((20, 189, 20, 255), (128, 50, 50, 255), (23, 21, 255, 255))
        for colorkey in colorkeys:
            s.set_colorkey(colorkey)
            for t in range(4):
                s.set_colorkey(s.get_colorkey())
            self.assertEqual(s.get_colorkey(), colorkey)

    def test_set_masks(self):
        s = pygame.Surface((32, 32))
        r, g, b, a = s.get_masks()
        self.assertRaises(TypeError, s.set_masks, (b, g, r, a))

    def test_set_shifts(self):
        s = pygame.Surface((32, 32))
        r, g, b, a = s.get_shifts()
        self.assertRaises(TypeError, s.set_shifts, (b, g, r, a))

    def test_blit_keyword_args(self):
        color = (1, 2, 3, 255)
        s1 = pygame.Surface((4, 4), 0, 32)
        s2 = pygame.Surface((2, 2), 0, 32)
        s2.fill((1, 2, 3))
        s1.blit(special_flags=BLEND_ADD, source=s2, dest=(1, 1), area=s2.get_rect())
        self.assertEqual(s1.get_at((0, 0)), (0, 0, 0, 255))
        self.assertEqual(s1.get_at((1, 1)), color)

    def test_blit_big_rects(self):
        """SDL2 can have more than 16 bits for x, y, width, height."""
        big_surf = pygame.Surface((100, 68000), 0, 32)
        big_surf_color = (255, 0, 0)
        big_surf.fill(big_surf_color)
        background = pygame.Surface((500, 500), 0, 32)
        background_color = (0, 255, 0)
        background.fill(background_color)
        background.blit(big_surf, (100, 100), area=(0, 16000, 100, 100))
        background.blit(big_surf, (200, 200), area=(0, 32000, 100, 100))
        background.blit(big_surf, (300, 300), area=(0, 66000, 100, 100))
        self.assertEqual(background.get_at((101, 101)), big_surf_color)
        self.assertEqual(background.get_at((201, 201)), big_surf_color)
        self.assertEqual(background.get_at((301, 301)), big_surf_color)
        self.assertEqual(background.get_at((400, 301)), background_color)
        self.assertEqual(background.get_at((400, 201)), background_color)
        self.assertEqual(background.get_at((100, 201)), background_color)
        self.assertEqual(background.get_at((99, 99)), background_color)
        self.assertEqual(background.get_at((450, 450)), background_color)