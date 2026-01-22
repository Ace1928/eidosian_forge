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
class GeneralSurfaceTests(unittest.TestCase):

    @unittest.skipIf(os.environ.get('SDL_VIDEODRIVER') == 'dummy', 'requires a non-"dummy" SDL_VIDEODRIVER')
    def test_image_convert_bug_131(self):
        pygame.display.init()
        try:
            pygame.display.set_mode((640, 480))
            im = pygame.image.load(example_path(os.path.join('data', 'city.png')))
            im2 = pygame.image.load(example_path(os.path.join('data', 'brick.png')))
            self.assertEqual(im.get_palette(), ((0, 0, 0, 255), (255, 255, 255, 255)))
            self.assertEqual(im2.get_palette(), ((0, 0, 0, 255), (0, 0, 0, 255)))
            self.assertEqual(repr(im.convert(32)), '<Surface(24x24x32 SW)>')
            self.assertEqual(repr(im2.convert(32)), '<Surface(469x137x32 SW)>')
            im3 = im.convert(8)
            self.assertEqual(repr(im3), '<Surface(24x24x8 SW)>')
            self.assertEqual(im3.get_palette(), im.get_palette())
        finally:
            pygame.display.quit()

    def test_convert_init(self):
        """Ensure initialization exceptions are raised
        for surf.convert()."""
        pygame.display.quit()
        surf = pygame.Surface((1, 1))
        self.assertRaisesRegex(pygame.error, 'display initialized', surf.convert)
        pygame.display.init()
        try:
            if os.environ.get('SDL_VIDEODRIVER') != 'dummy':
                try:
                    surf.convert(32)
                    surf.convert(pygame.Surface((1, 1)))
                except pygame.error:
                    self.fail('convert() should not raise an exception here.')
            self.assertRaisesRegex(pygame.error, 'No video mode', surf.convert)
            pygame.display.set_mode((640, 480))
            try:
                surf.convert()
            except pygame.error:
                self.fail('convert() should not raise an exception here.')
        finally:
            pygame.display.quit()

    def test_convert_alpha_init(self):
        """Ensure initialization exceptions are raised
        for surf.convert_alpha()."""
        pygame.display.quit()
        surf = pygame.Surface((1, 1))
        self.assertRaisesRegex(pygame.error, 'display initialized', surf.convert_alpha)
        pygame.display.init()
        try:
            self.assertRaisesRegex(pygame.error, 'No video mode', surf.convert_alpha)
            pygame.display.set_mode((640, 480))
            try:
                surf.convert_alpha()
            except pygame.error:
                self.fail('convert_alpha() should not raise an exception here.')
        finally:
            pygame.display.quit()

    def test_convert_alpha_SRCALPHA(self):
        """Ensure that the surface returned by surf.convert_alpha()
        has alpha blending enabled"""
        pygame.display.init()
        try:
            pygame.display.set_mode((640, 480))
            s1 = pygame.Surface((100, 100), 0, 32)
            s1_alpha = s1.convert_alpha()
            self.assertEqual(s1_alpha.get_flags() & SRCALPHA, SRCALPHA)
            self.assertEqual(s1_alpha.get_alpha(), 255)
        finally:
            pygame.display.quit()

    def test_src_alpha_issue_1289(self):
        """blit should be white."""
        surf1 = pygame.Surface((1, 1), pygame.SRCALPHA, 32)
        surf1.fill((255, 255, 255, 100))
        surf2 = pygame.Surface((1, 1), pygame.SRCALPHA, 32)
        self.assertEqual(surf2.get_at((0, 0)), (0, 0, 0, 0))
        surf2.blit(surf1, (0, 0))
        self.assertEqual(surf1.get_at((0, 0)), (255, 255, 255, 100))
        self.assertEqual(surf2.get_at((0, 0)), (255, 255, 255, 100))

    def test_src_alpha_compatible(self):
        """ "What pygame 1.9.x did". Is the alpha blitter as before?"""
        results_expected = {((0, 255, 255), (0, 255, 0)): (0, 255, 255, 255), ((0, 255, 255), (1, 254, 1)): (0, 255, 255, 255), ((0, 255, 255), (65, 199, 65)): (16, 255, 241, 255), ((0, 255, 255), (126, 127, 126)): (62, 255, 192, 255), ((0, 255, 255), (127, 126, 127)): (63, 255, 191, 255), ((0, 255, 255), (199, 65, 199)): (155, 255, 107, 255), ((0, 255, 255), (254, 1, 254)): (253, 255, 2, 255), ((0, 255, 255), (255, 0, 255)): (255, 255, 0, 255), ((1, 254, 254), (0, 255, 0)): (1, 255, 254, 254), ((1, 254, 254), (1, 254, 1)): (1, 255, 254, 255), ((1, 254, 254), (65, 199, 65)): (17, 255, 240, 255), ((1, 254, 254), (126, 127, 126)): (63, 255, 191, 255), ((1, 254, 254), (127, 126, 127)): (64, 255, 190, 255), ((1, 254, 254), (199, 65, 199)): (155, 255, 107, 255), ((1, 254, 254), (254, 1, 254)): (253, 255, 2, 255), ((1, 254, 254), (255, 0, 255)): (255, 255, 0, 255), ((65, 199, 199), (0, 255, 0)): (65, 255, 199, 199), ((65, 199, 199), (1, 254, 1)): (64, 255, 200, 200), ((65, 199, 199), (65, 199, 65)): (65, 255, 199, 214), ((65, 199, 199), (126, 127, 126)): (95, 255, 164, 227), ((65, 199, 199), (127, 126, 127)): (96, 255, 163, 227), ((65, 199, 199), (199, 65, 199)): (169, 255, 95, 243), ((65, 199, 199), (254, 1, 254)): (253, 255, 2, 255), ((65, 199, 199), (255, 0, 255)): (255, 255, 0, 255), ((126, 127, 127), (0, 255, 0)): (126, 255, 127, 127), ((126, 127, 127), (1, 254, 1)): (125, 255, 128, 128), ((126, 127, 127), (65, 199, 65)): (110, 255, 146, 160), ((126, 127, 127), (126, 127, 126)): (126, 255, 127, 191), ((126, 127, 127), (127, 126, 127)): (126, 255, 126, 191), ((126, 127, 127), (199, 65, 199)): (183, 255, 79, 227), ((126, 127, 127), (254, 1, 254)): (253, 255, 1, 255), ((126, 127, 127), (255, 0, 255)): (255, 255, 0, 255), ((127, 126, 126), (0, 255, 0)): (127, 255, 126, 126), ((127, 126, 126), (1, 254, 1)): (126, 255, 127, 127), ((127, 126, 126), (65, 199, 65)): (111, 255, 145, 159), ((127, 126, 126), (126, 127, 126)): (127, 255, 126, 190), ((127, 126, 126), (127, 126, 127)): (127, 255, 126, 191), ((127, 126, 126), (199, 65, 199)): (183, 255, 78, 227), ((127, 126, 126), (254, 1, 254)): (254, 255, 1, 255), ((127, 126, 126), (255, 0, 255)): (255, 255, 0, 255), ((199, 65, 65), (0, 255, 0)): (199, 255, 65, 65), ((199, 65, 65), (1, 254, 1)): (198, 255, 66, 66), ((199, 65, 65), (65, 199, 65)): (165, 255, 99, 114), ((199, 65, 65), (126, 127, 126)): (163, 255, 96, 159), ((199, 65, 65), (127, 126, 127)): (163, 255, 95, 160), ((199, 65, 65), (199, 65, 199)): (199, 255, 65, 214), ((199, 65, 65), (254, 1, 254)): (254, 255, 1, 255), ((199, 65, 65), (255, 0, 255)): (255, 255, 0, 255), ((254, 1, 1), (0, 255, 0)): (254, 255, 1, 1), ((254, 1, 1), (1, 254, 1)): (253, 255, 2, 2), ((254, 1, 1), (65, 199, 65)): (206, 255, 52, 66), ((254, 1, 1), (126, 127, 126)): (191, 255, 63, 127), ((254, 1, 1), (127, 126, 127)): (191, 255, 63, 128), ((254, 1, 1), (199, 65, 199)): (212, 255, 51, 200), ((254, 1, 1), (254, 1, 254)): (254, 255, 1, 255), ((254, 1, 1), (255, 0, 255)): (255, 255, 0, 255), ((255, 0, 0), (0, 255, 0)): (0, 255, 255, 0), ((255, 0, 0), (1, 254, 1)): (1, 255, 254, 1), ((255, 0, 0), (65, 199, 65)): (65, 255, 199, 65), ((255, 0, 0), (126, 127, 126)): (126, 255, 127, 126), ((255, 0, 0), (127, 126, 127)): (127, 255, 126, 127), ((255, 0, 0), (199, 65, 199)): (199, 255, 65, 199), ((255, 0, 0), (254, 1, 254)): (254, 255, 1, 254), ((255, 0, 0), (255, 0, 255)): (255, 255, 0, 255)}
        nums = [0, 1, 65, 126, 127, 199, 254, 255]
        results = {}
        for dst_r, dst_b, dst_a in zip(nums, reversed(nums), reversed(nums)):
            for src_r, src_b, src_a in zip(nums, reversed(nums), nums):
                with self.subTest(src_r=src_r, src_b=src_b, src_a=src_a, dest_r=dst_r, dest_b=dst_b, dest_a=dst_a):
                    src_surf = pygame.Surface((66, 66), pygame.SRCALPHA, 32)
                    src_surf.fill((src_r, 255, src_b, src_a))
                    dest_surf = pygame.Surface((66, 66), pygame.SRCALPHA, 32)
                    dest_surf.fill((dst_r, 255, dst_b, dst_a))
                    dest_surf.blit(src_surf, (0, 0))
                    key = ((dst_r, dst_b, dst_a), (src_r, src_b, src_a))
                    results[key] = dest_surf.get_at((65, 33))
                    self.assertEqual(results[key], results_expected[key])
        self.assertEqual(results, results_expected)

    def test_src_alpha_compatible_16bit(self):
        """ "What pygame 1.9.x did". Is the alpha blitter as before?"""
        results_expected = {((0, 255, 255), (0, 255, 0)): (0, 255, 255, 255), ((0, 255, 255), (1, 254, 1)): (0, 255, 255, 255), ((0, 255, 255), (65, 199, 65)): (17, 255, 255, 255), ((0, 255, 255), (126, 127, 126)): (51, 255, 204, 255), ((0, 255, 255), (127, 126, 127)): (51, 255, 204, 255), ((0, 255, 255), (199, 65, 199)): (170, 255, 102, 255), ((0, 255, 255), (254, 1, 254)): (255, 255, 0, 255), ((0, 255, 255), (255, 0, 255)): (255, 255, 0, 255), ((1, 254, 254), (0, 255, 0)): (0, 255, 255, 255), ((1, 254, 254), (1, 254, 1)): (0, 255, 255, 255), ((1, 254, 254), (65, 199, 65)): (17, 255, 255, 255), ((1, 254, 254), (126, 127, 126)): (51, 255, 204, 255), ((1, 254, 254), (127, 126, 127)): (51, 255, 204, 255), ((1, 254, 254), (199, 65, 199)): (170, 255, 102, 255), ((1, 254, 254), (254, 1, 254)): (255, 255, 0, 255), ((1, 254, 254), (255, 0, 255)): (255, 255, 0, 255), ((65, 199, 199), (0, 255, 0)): (68, 255, 204, 204), ((65, 199, 199), (1, 254, 1)): (68, 255, 204, 204), ((65, 199, 199), (65, 199, 65)): (68, 255, 204, 221), ((65, 199, 199), (126, 127, 126)): (85, 255, 170, 238), ((65, 199, 199), (127, 126, 127)): (85, 255, 170, 238), ((65, 199, 199), (199, 65, 199)): (187, 255, 85, 255), ((65, 199, 199), (254, 1, 254)): (255, 255, 0, 255), ((65, 199, 199), (255, 0, 255)): (255, 255, 0, 255), ((126, 127, 127), (0, 255, 0)): (119, 255, 119, 119), ((126, 127, 127), (1, 254, 1)): (119, 255, 119, 119), ((126, 127, 127), (65, 199, 65)): (102, 255, 136, 153), ((126, 127, 127), (126, 127, 126)): (119, 255, 119, 187), ((126, 127, 127), (127, 126, 127)): (119, 255, 119, 187), ((126, 127, 127), (199, 65, 199)): (187, 255, 68, 238), ((126, 127, 127), (254, 1, 254)): (255, 255, 0, 255), ((126, 127, 127), (255, 0, 255)): (255, 255, 0, 255), ((127, 126, 126), (0, 255, 0)): (119, 255, 119, 119), ((127, 126, 126), (1, 254, 1)): (119, 255, 119, 119), ((127, 126, 126), (65, 199, 65)): (102, 255, 136, 153), ((127, 126, 126), (126, 127, 126)): (119, 255, 119, 187), ((127, 126, 126), (127, 126, 127)): (119, 255, 119, 187), ((127, 126, 126), (199, 65, 199)): (187, 255, 68, 238), ((127, 126, 126), (254, 1, 254)): (255, 255, 0, 255), ((127, 126, 126), (255, 0, 255)): (255, 255, 0, 255), ((199, 65, 65), (0, 255, 0)): (204, 255, 68, 68), ((199, 65, 65), (1, 254, 1)): (204, 255, 68, 68), ((199, 65, 65), (65, 199, 65)): (170, 255, 102, 119), ((199, 65, 65), (126, 127, 126)): (170, 255, 85, 153), ((199, 65, 65), (127, 126, 127)): (170, 255, 85, 153), ((199, 65, 65), (199, 65, 199)): (204, 255, 68, 221), ((199, 65, 65), (254, 1, 254)): (255, 255, 0, 255), ((199, 65, 65), (255, 0, 255)): (255, 255, 0, 255), ((254, 1, 1), (0, 255, 0)): (0, 255, 255, 0), ((254, 1, 1), (1, 254, 1)): (0, 255, 255, 0), ((254, 1, 1), (65, 199, 65)): (68, 255, 204, 68), ((254, 1, 1), (126, 127, 126)): (119, 255, 119, 119), ((254, 1, 1), (127, 126, 127)): (119, 255, 119, 119), ((254, 1, 1), (199, 65, 199)): (204, 255, 68, 204), ((254, 1, 1), (254, 1, 254)): (255, 255, 0, 255), ((254, 1, 1), (255, 0, 255)): (255, 255, 0, 255), ((255, 0, 0), (0, 255, 0)): (0, 255, 255, 0), ((255, 0, 0), (1, 254, 1)): (0, 255, 255, 0), ((255, 0, 0), (65, 199, 65)): (68, 255, 204, 68), ((255, 0, 0), (126, 127, 126)): (119, 255, 119, 119), ((255, 0, 0), (127, 126, 127)): (119, 255, 119, 119), ((255, 0, 0), (199, 65, 199)): (204, 255, 68, 204), ((255, 0, 0), (254, 1, 254)): (255, 255, 0, 255), ((255, 0, 0), (255, 0, 255)): (255, 255, 0, 255)}
        nums = [0, 1, 65, 126, 127, 199, 254, 255]
        results = {}
        for dst_r, dst_b, dst_a in zip(nums, reversed(nums), reversed(nums)):
            for src_r, src_b, src_a in zip(nums, reversed(nums), nums):
                with self.subTest(src_r=src_r, src_b=src_b, src_a=src_a, dest_r=dst_r, dest_b=dst_b, dest_a=dst_a):
                    src_surf = pygame.Surface((66, 66), pygame.SRCALPHA, 16)
                    src_surf.fill((src_r, 255, src_b, src_a))
                    dest_surf = pygame.Surface((66, 66), pygame.SRCALPHA, 16)
                    dest_surf.fill((dst_r, 255, dst_b, dst_a))
                    dest_surf.blit(src_surf, (0, 0))
                    key = ((dst_r, dst_b, dst_a), (src_r, src_b, src_a))
                    results[key] = dest_surf.get_at((65, 33))
                    self.assertEqual(results[key], results_expected[key])
        self.assertEqual(results, results_expected)

    def test_sdl1_mimic_blitter_with_set_alpha(self):
        """does the SDL 1 style blitter in pygame 2 work with set_alpha(),
        this feature only exists in pygame 2/SDL2 SDL1 did not support
        combining surface and pixel alpha"""
        results_expected = {((0, 255, 255), (0, 255, 0)): (0, 255, 255, 255), ((0, 255, 255), (1, 254, 1)): (0, 255, 255, 255), ((0, 255, 255), (65, 199, 65)): (16, 255, 241, 255), ((0, 255, 255), (126, 127, 126)): (62, 255, 192, 255), ((0, 255, 255), (127, 126, 127)): (63, 255, 191, 255), ((0, 255, 255), (199, 65, 199)): (155, 255, 107, 255), ((0, 255, 255), (254, 1, 254)): (253, 255, 2, 255), ((0, 255, 255), (255, 0, 255)): (255, 255, 0, 255), ((1, 254, 254), (0, 255, 0)): (1, 255, 254, 254), ((1, 254, 254), (1, 254, 1)): (1, 255, 254, 255), ((1, 254, 254), (65, 199, 65)): (17, 255, 240, 255), ((1, 254, 254), (126, 127, 126)): (63, 255, 191, 255), ((1, 254, 254), (127, 126, 127)): (64, 255, 190, 255), ((1, 254, 254), (199, 65, 199)): (155, 255, 107, 255), ((1, 254, 254), (254, 1, 254)): (253, 255, 2, 255), ((1, 254, 254), (255, 0, 255)): (255, 255, 0, 255), ((65, 199, 199), (0, 255, 0)): (65, 255, 199, 199), ((65, 199, 199), (1, 254, 1)): (64, 255, 200, 200), ((65, 199, 199), (65, 199, 65)): (65, 255, 199, 214), ((65, 199, 199), (126, 127, 126)): (95, 255, 164, 227), ((65, 199, 199), (127, 126, 127)): (96, 255, 163, 227), ((65, 199, 199), (199, 65, 199)): (169, 255, 95, 243), ((65, 199, 199), (254, 1, 254)): (253, 255, 2, 255), ((65, 199, 199), (255, 0, 255)): (255, 255, 0, 255), ((126, 127, 127), (0, 255, 0)): (126, 255, 127, 127), ((126, 127, 127), (1, 254, 1)): (125, 255, 128, 128), ((126, 127, 127), (65, 199, 65)): (110, 255, 146, 160), ((126, 127, 127), (126, 127, 126)): (126, 255, 127, 191), ((126, 127, 127), (127, 126, 127)): (126, 255, 126, 191), ((126, 127, 127), (199, 65, 199)): (183, 255, 79, 227), ((126, 127, 127), (254, 1, 254)): (253, 255, 1, 255), ((126, 127, 127), (255, 0, 255)): (255, 255, 0, 255), ((127, 126, 126), (0, 255, 0)): (127, 255, 126, 126), ((127, 126, 126), (1, 254, 1)): (126, 255, 127, 127), ((127, 126, 126), (65, 199, 65)): (111, 255, 145, 159), ((127, 126, 126), (126, 127, 126)): (127, 255, 126, 190), ((127, 126, 126), (127, 126, 127)): (127, 255, 126, 191), ((127, 126, 126), (199, 65, 199)): (183, 255, 78, 227), ((127, 126, 126), (254, 1, 254)): (254, 255, 1, 255), ((127, 126, 126), (255, 0, 255)): (255, 255, 0, 255), ((199, 65, 65), (0, 255, 0)): (199, 255, 65, 65), ((199, 65, 65), (1, 254, 1)): (198, 255, 66, 66), ((199, 65, 65), (65, 199, 65)): (165, 255, 99, 114), ((199, 65, 65), (126, 127, 126)): (163, 255, 96, 159), ((199, 65, 65), (127, 126, 127)): (163, 255, 95, 160), ((199, 65, 65), (199, 65, 199)): (199, 255, 65, 214), ((199, 65, 65), (254, 1, 254)): (254, 255, 1, 255), ((199, 65, 65), (255, 0, 255)): (255, 255, 0, 255), ((254, 1, 1), (0, 255, 0)): (254, 255, 1, 1), ((254, 1, 1), (1, 254, 1)): (253, 255, 2, 2), ((254, 1, 1), (65, 199, 65)): (206, 255, 52, 66), ((254, 1, 1), (126, 127, 126)): (191, 255, 63, 127), ((254, 1, 1), (127, 126, 127)): (191, 255, 63, 128), ((254, 1, 1), (199, 65, 199)): (212, 255, 51, 200), ((254, 1, 1), (254, 1, 254)): (254, 255, 1, 255), ((254, 1, 1), (255, 0, 255)): (255, 255, 0, 255), ((255, 0, 0), (0, 255, 0)): (0, 255, 255, 0), ((255, 0, 0), (1, 254, 1)): (1, 255, 254, 1), ((255, 0, 0), (65, 199, 65)): (65, 255, 199, 65), ((255, 0, 0), (126, 127, 126)): (126, 255, 127, 126), ((255, 0, 0), (127, 126, 127)): (127, 255, 126, 127), ((255, 0, 0), (199, 65, 199)): (199, 255, 65, 199), ((255, 0, 0), (254, 1, 254)): (254, 255, 1, 254), ((255, 0, 0), (255, 0, 255)): (255, 255, 0, 255)}
        nums = [0, 1, 65, 126, 127, 199, 254, 255]
        results = {}
        for dst_r, dst_b, dst_a in zip(nums, reversed(nums), reversed(nums)):
            for src_r, src_b, src_a in zip(nums, reversed(nums), nums):
                with self.subTest(src_r=src_r, src_b=src_b, src_a=src_a, dest_r=dst_r, dest_b=dst_b, dest_a=dst_a):
                    src_surf = pygame.Surface((66, 66), pygame.SRCALPHA, 32)
                    src_surf.fill((src_r, 255, src_b, 255))
                    src_surf.set_alpha(src_a)
                    dest_surf = pygame.Surface((66, 66), pygame.SRCALPHA, 32)
                    dest_surf.fill((dst_r, 255, dst_b, dst_a))
                    dest_surf.blit(src_surf, (0, 0))
                    key = ((dst_r, dst_b, dst_a), (src_r, src_b, src_a))
                    results[key] = dest_surf.get_at((65, 33))
                    self.assertEqual(results[key], results_expected[key])
        self.assertEqual(results, results_expected)

    @unittest.skipIf('arm' in platform.machine() or 'aarch64' in platform.machine(), 'sdl2 blitter produces different results on arm')
    def test_src_alpha_sdl2_blitter(self):
        """Checking that the BLEND_ALPHA_SDL2 flag works - this feature
        only exists when using SDL2"""
        results_expected = {((0, 255, 255), (0, 255, 0)): (0, 255, 255, 255), ((0, 255, 255), (1, 254, 1)): (0, 253, 253, 253), ((0, 255, 255), (65, 199, 65)): (16, 253, 239, 253), ((0, 255, 255), (126, 127, 126)): (62, 253, 190, 253), ((0, 255, 255), (127, 126, 127)): (63, 253, 189, 253), ((0, 255, 255), (199, 65, 199)): (154, 253, 105, 253), ((0, 255, 255), (254, 1, 254)): (252, 253, 0, 253), ((0, 255, 255), (255, 0, 255)): (255, 255, 0, 255), ((1, 254, 254), (0, 255, 0)): (1, 255, 254, 254), ((1, 254, 254), (1, 254, 1)): (0, 253, 252, 252), ((1, 254, 254), (65, 199, 65)): (16, 253, 238, 252), ((1, 254, 254), (126, 127, 126)): (62, 253, 189, 252), ((1, 254, 254), (127, 126, 127)): (63, 253, 189, 253), ((1, 254, 254), (199, 65, 199)): (154, 253, 105, 253), ((1, 254, 254), (254, 1, 254)): (252, 253, 0, 253), ((1, 254, 254), (255, 0, 255)): (255, 255, 0, 255), ((65, 199, 199), (0, 255, 0)): (65, 255, 199, 199), ((65, 199, 199), (1, 254, 1)): (64, 253, 197, 197), ((65, 199, 199), (65, 199, 65)): (64, 253, 197, 211), ((65, 199, 199), (126, 127, 126)): (94, 253, 162, 225), ((65, 199, 199), (127, 126, 127)): (95, 253, 161, 225), ((65, 199, 199), (199, 65, 199)): (168, 253, 93, 241), ((65, 199, 199), (254, 1, 254)): (252, 253, 0, 253), ((65, 199, 199), (255, 0, 255)): (255, 255, 0, 255), ((126, 127, 127), (0, 255, 0)): (126, 255, 127, 127), ((126, 127, 127), (1, 254, 1)): (125, 253, 126, 126), ((126, 127, 127), (65, 199, 65)): (109, 253, 144, 158), ((126, 127, 127), (126, 127, 126)): (125, 253, 125, 188), ((126, 127, 127), (127, 126, 127)): (126, 253, 125, 189), ((126, 127, 127), (199, 65, 199)): (181, 253, 77, 225), ((126, 127, 127), (254, 1, 254)): (252, 253, 0, 253), ((126, 127, 127), (255, 0, 255)): (255, 255, 0, 255), ((127, 126, 126), (0, 255, 0)): (127, 255, 126, 126), ((127, 126, 126), (1, 254, 1)): (126, 253, 125, 125), ((127, 126, 126), (65, 199, 65)): (110, 253, 143, 157), ((127, 126, 126), (126, 127, 126)): (125, 253, 125, 188), ((127, 126, 126), (127, 126, 127)): (126, 253, 125, 189), ((127, 126, 126), (199, 65, 199)): (181, 253, 77, 225), ((127, 126, 126), (254, 1, 254)): (252, 253, 0, 253), ((127, 126, 126), (255, 0, 255)): (255, 255, 0, 255), ((199, 65, 65), (0, 255, 0)): (199, 255, 65, 65), ((199, 65, 65), (1, 254, 1)): (197, 253, 64, 64), ((199, 65, 65), (65, 199, 65)): (163, 253, 98, 112), ((199, 65, 65), (126, 127, 126)): (162, 253, 94, 157), ((199, 65, 65), (127, 126, 127)): (162, 253, 94, 158), ((199, 65, 65), (199, 65, 199)): (197, 253, 64, 212), ((199, 65, 65), (254, 1, 254)): (252, 253, 0, 253), ((199, 65, 65), (255, 0, 255)): (255, 255, 0, 255), ((254, 1, 1), (0, 255, 0)): (254, 255, 1, 1), ((254, 1, 1), (1, 254, 1)): (252, 253, 0, 0), ((254, 1, 1), (65, 199, 65)): (204, 253, 50, 64), ((254, 1, 1), (126, 127, 126)): (189, 253, 62, 125), ((254, 1, 1), (127, 126, 127)): (190, 253, 62, 126), ((254, 1, 1), (199, 65, 199)): (209, 253, 50, 198), ((254, 1, 1), (254, 1, 254)): (252, 253, 0, 253), ((254, 1, 1), (255, 0, 255)): (255, 255, 0, 255), ((255, 0, 0), (0, 255, 0)): (255, 255, 0, 0), ((255, 0, 0), (1, 254, 1)): (253, 253, 0, 0), ((255, 0, 0), (65, 199, 65)): (205, 253, 50, 64), ((255, 0, 0), (126, 127, 126)): (190, 253, 62, 125), ((255, 0, 0), (127, 126, 127)): (190, 253, 62, 126), ((255, 0, 0), (199, 65, 199)): (209, 253, 50, 198), ((255, 0, 0), (254, 1, 254)): (252, 253, 0, 253), ((255, 0, 0), (255, 0, 255)): (255, 255, 0, 255)}
        nums = [0, 1, 65, 126, 127, 199, 254, 255]
        results = {}
        for dst_r, dst_b, dst_a in zip(nums, reversed(nums), reversed(nums)):
            for src_r, src_b, src_a in zip(nums, reversed(nums), nums):
                with self.subTest(src_r=src_r, src_b=src_b, src_a=src_a, dest_r=dst_r, dest_b=dst_b, dest_a=dst_a):
                    src_surf = pygame.Surface((66, 66), pygame.SRCALPHA, 32)
                    src_surf.fill((src_r, 255, src_b, src_a))
                    dest_surf = pygame.Surface((66, 66), pygame.SRCALPHA, 32)
                    dest_surf.fill((dst_r, 255, dst_b, dst_a))
                    dest_surf.blit(src_surf, (0, 0), special_flags=pygame.BLEND_ALPHA_SDL2)
                    key = ((dst_r, dst_b, dst_a), (src_r, src_b, src_a))
                    results[key] = tuple(dest_surf.get_at((65, 33)))
                    for i in range(4):
                        self.assertAlmostEqual(results[key][i], results_expected[key][i], delta=4)

    def test_opaque_destination_blit_with_set_alpha(self):
        src_surf = pygame.Surface((32, 32), pygame.SRCALPHA, 32)
        src_surf.fill((255, 255, 255, 200))
        dest_surf = pygame.Surface((32, 32))
        dest_surf.fill((100, 100, 100))
        dest_surf.blit(src_surf, (0, 0))
        no_surf_alpha_col = dest_surf.get_at((0, 0))
        dest_surf.fill((100, 100, 100))
        dest_surf.set_alpha(200)
        dest_surf.blit(src_surf, (0, 0))
        surf_alpha_col = dest_surf.get_at((0, 0))
        self.assertEqual(no_surf_alpha_col, surf_alpha_col)

    def todo_test_convert(self):
        self.fail()

    def test_convert__pixel_format_as_surface_subclass(self):
        """Ensure convert accepts a Surface subclass argument."""
        expected_size = (23, 17)
        convert_surface = SurfaceSubclass(expected_size, 0, 32)
        depth_surface = SurfaceSubclass((31, 61), 0, 32)
        pygame.display.init()
        try:
            surface = convert_surface.convert(depth_surface)
            self.assertIsNot(surface, depth_surface)
            self.assertIsNot(surface, convert_surface)
            self.assertIsInstance(surface, pygame.Surface)
            self.assertIsInstance(surface, SurfaceSubclass)
            self.assertEqual(surface.get_size(), expected_size)
        finally:
            pygame.display.quit()

    def test_convert_alpha(self):
        """Ensure the surface returned by surf.convert_alpha
        has alpha values added"""
        pygame.display.init()
        try:
            pygame.display.set_mode((640, 480))
            s1 = pygame.Surface((100, 100), 0, 32)
            s1_alpha = pygame.Surface.convert_alpha(s1)
            s2 = pygame.Surface((100, 100), 0, 32)
            s2_alpha = s2.convert_alpha()
            s3 = pygame.Surface((100, 100), 0, 8)
            s3_alpha = s3.convert_alpha()
            s4 = pygame.Surface((100, 100), 0, 12)
            s4_alpha = s4.convert_alpha()
            s5 = pygame.Surface((100, 100), 0, 15)
            s5_alpha = s5.convert_alpha()
            s6 = pygame.Surface((100, 100), 0, 16)
            s6_alpha = s6.convert_alpha()
            s7 = pygame.Surface((100, 100), 0, 24)
            s7_alpha = s7.convert_alpha()
            self.assertEqual(s1_alpha.get_alpha(), 255)
            self.assertEqual(s2_alpha.get_alpha(), 255)
            self.assertEqual(s3_alpha.get_alpha(), 255)
            self.assertEqual(s4_alpha.get_alpha(), 255)
            self.assertEqual(s5_alpha.get_alpha(), 255)
            self.assertEqual(s6_alpha.get_alpha(), 255)
            self.assertEqual(s7_alpha.get_alpha(), 255)
            self.assertEqual(s1_alpha.get_bitsize(), 32)
            self.assertEqual(s2_alpha.get_bitsize(), 32)
            self.assertEqual(s3_alpha.get_bitsize(), 32)
            self.assertEqual(s4_alpha.get_bitsize(), 32)
            self.assertEqual(s5_alpha.get_bitsize(), 32)
            self.assertEqual(s6_alpha.get_bitsize(), 32)
            self.assertEqual(s6_alpha.get_bitsize(), 32)
            with self.assertRaises(pygame.error):
                surface = pygame.display.set_mode()
                pygame.display.quit()
                surface.convert_alpha()
        finally:
            pygame.display.quit()

    def test_convert_alpha__pixel_format_as_surface_subclass(self):
        """Ensure convert_alpha accepts a Surface subclass argument."""
        expected_size = (23, 17)
        convert_surface = SurfaceSubclass(expected_size, SRCALPHA, 32)
        depth_surface = SurfaceSubclass((31, 57), SRCALPHA, 32)
        pygame.display.init()
        try:
            pygame.display.set_mode((60, 60))
            surface = convert_surface.convert_alpha(depth_surface)
            self.assertIsNot(surface, depth_surface)
            self.assertIsNot(surface, convert_surface)
            self.assertIsInstance(surface, pygame.Surface)
            self.assertIsInstance(surface, SurfaceSubclass)
            self.assertEqual(surface.get_size(), expected_size)
        finally:
            pygame.display.quit()

    def test_get_abs_offset(self):
        pygame.display.init()
        try:
            parent = pygame.Surface((64, 64), SRCALPHA, 32)
            sub_level_1 = parent.subsurface((2, 2), (34, 37))
            sub_level_2 = sub_level_1.subsurface((0, 0), (30, 29))
            sub_level_3 = sub_level_2.subsurface((3, 7), (20, 21))
            sub_level_4 = sub_level_3.subsurface((6, 1), (14, 14))
            sub_level_5 = sub_level_4.subsurface((5, 6), (3, 4))
            self.assertEqual(parent.get_abs_offset(), (0, 0))
            self.assertEqual(sub_level_1.get_abs_offset(), (2, 2))
            self.assertEqual(sub_level_2.get_abs_offset(), (2, 2))
            self.assertEqual(sub_level_3.get_abs_offset(), (5, 9))
            self.assertEqual(sub_level_4.get_abs_offset(), (11, 10))
            self.assertEqual(sub_level_5.get_abs_offset(), (16, 16))
            with self.assertRaises(pygame.error):
                surface = pygame.display.set_mode()
                pygame.display.quit()
                surface.get_abs_offset()
        finally:
            pygame.display.quit()

    def test_get_abs_parent(self):
        pygame.display.init()
        try:
            parent = pygame.Surface((32, 32), SRCALPHA, 32)
            sub_level_1 = parent.subsurface((1, 1), (15, 15))
            sub_level_2 = sub_level_1.subsurface((1, 1), (12, 12))
            sub_level_3 = sub_level_2.subsurface((1, 1), (9, 9))
            sub_level_4 = sub_level_3.subsurface((1, 1), (8, 8))
            sub_level_5 = sub_level_4.subsurface((2, 2), (3, 4))
            sub_level_6 = sub_level_5.subsurface((0, 0), (2, 1))
            self.assertRaises(ValueError, parent.subsurface, (5, 5), (100, 100))
            self.assertRaises(ValueError, sub_level_3.subsurface, (0, 0), (11, 5))
            self.assertRaises(ValueError, sub_level_6.subsurface, (0, 0), (5, 5))
            self.assertEqual(parent.get_abs_parent(), parent)
            self.assertEqual(sub_level_1.get_abs_parent(), sub_level_1.get_parent())
            self.assertEqual(sub_level_2.get_abs_parent(), parent)
            self.assertEqual(sub_level_3.get_abs_parent(), parent)
            self.assertEqual(sub_level_4.get_abs_parent(), parent)
            self.assertEqual(sub_level_5.get_abs_parent(), parent)
            self.assertEqual(sub_level_6.get_abs_parent(), sub_level_6.get_parent().get_abs_parent())
            with self.assertRaises(pygame.error):
                surface = pygame.display.set_mode()
                pygame.display.quit()
                surface.get_abs_parent()
        finally:
            pygame.display.quit()

    def test_get_at(self):
        surf = pygame.Surface((2, 2), 0, 24)
        c00 = pygame.Color(1, 2, 3)
        c01 = pygame.Color(5, 10, 15)
        c10 = pygame.Color(100, 50, 0)
        c11 = pygame.Color(4, 5, 6)
        surf.set_at((0, 0), c00)
        surf.set_at((0, 1), c01)
        surf.set_at((1, 0), c10)
        surf.set_at((1, 1), c11)
        c = surf.get_at((0, 0))
        self.assertIsInstance(c, pygame.Color)
        self.assertEqual(c, c00)
        self.assertEqual(surf.get_at((0, 1)), c01)
        self.assertEqual(surf.get_at((1, 0)), c10)
        self.assertEqual(surf.get_at((1, 1)), c11)
        for p in [(-1, 0), (0, -1), (2, 0), (0, 2)]:
            self.assertRaises(IndexError, surf.get_at, p)

    def test_get_at_mapped(self):
        color = pygame.Color(10, 20, 30)
        for bitsize in [8, 16, 24, 32]:
            surf = pygame.Surface((2, 2), 0, bitsize)
            surf.fill(color)
            pixel = surf.get_at_mapped((0, 0))
            self.assertEqual(pixel, surf.map_rgb(color), '%i != %i, bitsize: %i' % (pixel, surf.map_rgb(color), bitsize))

    def test_get_bitsize(self):
        pygame.display.init()
        try:
            expected_size = (11, 21)
            expected_depth = 32
            surface = pygame.Surface(expected_size, pygame.SRCALPHA, expected_depth)
            self.assertEqual(surface.get_size(), expected_size)
            self.assertEqual(surface.get_bitsize(), expected_depth)
            expected_depth = 16
            surface = pygame.Surface(expected_size, pygame.SRCALPHA, expected_depth)
            self.assertEqual(surface.get_size(), expected_size)
            self.assertEqual(surface.get_bitsize(), expected_depth)
            expected_depth = 15
            surface = pygame.Surface(expected_size, 0, expected_depth)
            self.assertEqual(surface.get_size(), expected_size)
            self.assertEqual(surface.get_bitsize(), expected_depth)
            expected_depth = -1
            self.assertRaises(ValueError, pygame.Surface, expected_size, 0, expected_depth)
            expected_depth = 11
            self.assertRaises(ValueError, pygame.Surface, expected_size, 0, expected_depth)
            expected_depth = 1024
            self.assertRaises(ValueError, pygame.Surface, expected_size, 0, expected_depth)
            with self.assertRaises(pygame.error):
                surface = pygame.display.set_mode()
                pygame.display.quit()
                surface.get_bitsize()
        finally:
            pygame.display.quit()

    def test_get_clip(self):
        s = pygame.Surface((800, 600))
        rectangle = s.get_clip()
        self.assertEqual(rectangle, (0, 0, 800, 600))

    def test_get_colorkey(self):
        pygame.display.init()
        try:
            s = pygame.Surface((800, 600), 0, 32)
            self.assertIsNone(s.get_colorkey())
            s.set_colorkey(None)
            self.assertIsNone(s.get_colorkey())
            r, g, b, a = (20, 40, 60, 12)
            colorkey = pygame.Color(r, g, b)
            s.set_colorkey(colorkey)
            self.assertEqual(s.get_colorkey(), (r, g, b, 255))
            s.set_colorkey(colorkey, pygame.RLEACCEL)
            self.assertEqual(s.get_colorkey(), (r, g, b, 255))
            s.set_colorkey(pygame.Color(r + 1, g + 1, b + 1))
            self.assertNotEqual(s.get_colorkey(), (r, g, b, 255))
            s.set_colorkey(pygame.Color(r, g, b, a))
            self.assertEqual(s.get_colorkey(), (r, g, b, 255))
        finally:
            s = pygame.display.set_mode((200, 200), 0, 32)
            pygame.display.quit()
            with self.assertRaises(pygame.error):
                s.get_colorkey()

    def test_get_height(self):
        sizes = ((1, 1), (119, 10), (10, 119), (1, 1000), (1000, 1), (1000, 1000))
        for width, height in sizes:
            surf = pygame.Surface((width, height))
            found_height = surf.get_height()
            self.assertEqual(height, found_height)

    def test_get_locked(self):

        def blit_locked_test(surface):
            newSurf = pygame.Surface((10, 10))
            try:
                newSurf.blit(surface, (0, 0))
            except pygame.error:
                return True
            else:
                return False
        surf = pygame.Surface((100, 100))
        self.assertIs(surf.get_locked(), blit_locked_test(surf))
        surf.lock()
        self.assertIs(surf.get_locked(), blit_locked_test(surf))
        surf.unlock()
        self.assertIs(surf.get_locked(), blit_locked_test(surf))
        surf = pygame.Surface((100, 100))
        surf.lock()
        surf.lock()
        self.assertIs(surf.get_locked(), blit_locked_test(surf))
        surf.unlock()
        self.assertIs(surf.get_locked(), blit_locked_test(surf))
        surf.unlock()
        self.assertIs(surf.get_locked(), blit_locked_test(surf))
        surf = pygame.Surface((100, 100))
        for i in range(1000):
            surf.lock()
        self.assertIs(surf.get_locked(), blit_locked_test(surf))
        for i in range(1000):
            surf.unlock()
        self.assertFalse(surf.get_locked())
        surf = pygame.Surface((100, 100))
        surf.unlock()
        self.assertIs(surf.get_locked(), blit_locked_test(surf))
        surf.unlock()
        self.assertIs(surf.get_locked(), blit_locked_test(surf))

    def test_get_locks(self):
        surface = pygame.Surface((100, 100))
        self.assertEqual(surface.get_locks(), ())
        surface.lock()
        self.assertEqual(surface.get_locks(), (surface,))
        surface.unlock()
        self.assertEqual(surface.get_locks(), ())
        pxarray = pygame.PixelArray(surface)
        self.assertNotEqual(surface.get_locks(), ())
        pxarray.close()
        self.assertEqual(surface.get_locks(), ())
        with self.assertRaises(AttributeError):
            'DUMMY'.get_locks()
        surface.lock()
        surface.lock()
        surface.lock()
        self.assertEqual(surface.get_locks(), (surface, surface, surface))
        surface.unlock()
        surface.unlock()
        self.assertEqual(surface.get_locks(), (surface,))
        surface.unlock()
        self.assertEqual(surface.get_locks(), ())

    def test_get_losses(self):
        """Ensure a surface's losses can be retrieved"""
        pygame.display.init()
        try:
            mask8 = (224, 28, 3, 0)
            mask15 = (31744, 992, 31, 0)
            mask16 = (63488, 2016, 31, 0)
            mask24 = (16711680, 65280, 255, 0)
            mask32 = (4278190080, 16711680, 65280, 255)
            display_surf = pygame.display.set_mode((100, 100))
            surf = pygame.Surface((100, 100))
            surf_8bit = pygame.Surface((100, 100), depth=8, masks=mask8)
            surf_15bit = pygame.Surface((100, 100), depth=15, masks=mask15)
            surf_16bit = pygame.Surface((100, 100), depth=16, masks=mask16)
            surf_24bit = pygame.Surface((100, 100), depth=24, masks=mask24)
            surf_32bit = pygame.Surface((100, 100), depth=32, masks=mask32)
            losses = surf.get_losses()
            self.assertIsInstance(losses, tuple)
            self.assertEqual(len(losses), 4)
            for loss in losses:
                self.assertIsInstance(loss, int)
                self.assertGreaterEqual(loss, 0)
                self.assertLessEqual(loss, 8)
            if display_surf.get_losses() == (0, 0, 0, 8):
                self.assertEqual(losses, (0, 0, 0, 8))
            elif display_surf.get_losses() == (8, 8, 8, 8):
                self.assertEqual(losses, (8, 8, 8, 8))
            self.assertEqual(surf_8bit.get_losses(), (5, 5, 6, 8))
            self.assertEqual(surf_15bit.get_losses(), (3, 3, 3, 8))
            self.assertEqual(surf_16bit.get_losses(), (3, 2, 3, 8))
            self.assertEqual(surf_24bit.get_losses(), (0, 0, 0, 8))
            self.assertEqual(surf_32bit.get_losses(), (0, 0, 0, 0))
            with self.assertRaises(pygame.error):
                surface = pygame.display.set_mode((100, 100))
                pygame.display.quit()
                surface.get_losses()
        finally:
            pygame.display.quit()

    def test_get_masks__rgba(self):
        """
        Ensure that get_mask can return RGBA mask.
        """
        masks = [(3840, 240, 15, 61440), (16711680, 65280, 255, 4278190080)]
        depths = [16, 32]
        for expected, depth in list(zip(masks, depths)):
            surface = pygame.Surface((10, 10), pygame.SRCALPHA, depth)
            self.assertEqual(expected, surface.get_masks())

    def test_get_masks__rgb(self):
        """
        Ensure that get_mask can return RGB mask.
        """
        masks = [(96, 28, 3, 0), (3840, 240, 15, 0), (31744, 992, 31, 0), (63488, 2016, 31, 0), (16711680, 65280, 255, 0), (16711680, 65280, 255, 0)]
        depths = [8, 12, 15, 16, 24, 32]
        for expected, depth in list(zip(masks, depths)):
            surface = pygame.Surface((10, 10), 0, depth)
            if depth == 8:
                expected = (0, 0, 0, 0)
            self.assertEqual(expected, surface.get_masks())

    def test_get_masks__no_surface(self):
        """
        Ensure that after display.quit, calling get_masks raises pygame.error.
        """
        with self.assertRaises(pygame.error):
            surface = pygame.display.set_mode((10, 10))
            pygame.display.quit()
            surface.get_masks()

    def test_get_offset(self):
        """get_offset returns the (0,0) if surface is not a child
        returns the position of child subsurface inside of parent
        """
        pygame.display.init()
        try:
            surf = pygame.Surface((100, 100))
            self.assertEqual(surf.get_offset(), (0, 0))
            subsurf = surf.subsurface(1, 1, 10, 10)
            self.assertEqual(subsurf.get_offset(), (1, 1))
            with self.assertRaises(pygame.error):
                surface = pygame.display.set_mode()
                pygame.display.quit()
                surface.get_offset()
        finally:
            pygame.display.quit()

    def test_get_palette(self):
        palette = [Color(i, i, i) for i in range(256)]
        surf = pygame.Surface((2, 2), 0, 8)
        surf.set_palette(palette)
        palette2 = surf.get_palette()
        self.assertEqual(len(palette2), len(palette))
        for c2, c in zip(palette2, palette):
            self.assertEqual(c2, c)
        for c in palette2:
            self.assertIsInstance(c, pygame.Color)

    def test_get_palette_at(self):
        surf = pygame.Surface((2, 2), 0, 8)
        color = pygame.Color(1, 2, 3, 255)
        surf.set_palette_at(0, color)
        color2 = surf.get_palette_at(0)
        self.assertIsInstance(color2, pygame.Color)
        self.assertEqual(color2, color)
        self.assertRaises(IndexError, surf.get_palette_at, -1)
        self.assertRaises(IndexError, surf.get_palette_at, 256)

    def test_get_pitch(self):
        sizes = ((2, 2), (7, 33), (33, 7), (2, 734), (734, 2), (734, 734))
        depths = [8, 24, 32]
        for width, height in sizes:
            for depth in depths:
                surf = pygame.Surface((width, height), depth=depth)
                buff = surf.get_buffer()
                pitch = buff.length / surf.get_height()
                test_pitch = surf.get_pitch()
                self.assertEqual(pitch, test_pitch)
                rect1 = surf.get_rect()
                subsurf1 = surf.subsurface(rect1)
                sub_buff1 = subsurf1.get_buffer()
                sub_pitch1 = sub_buff1.length / subsurf1.get_height()
                test_sub_pitch1 = subsurf1.get_pitch()
                self.assertEqual(sub_pitch1, test_sub_pitch1)
                rect2 = rect1.inflate(-width / 2, -height / 2)
                subsurf2 = surf.subsurface(rect2)
                sub_buff2 = subsurf2.get_buffer()
                sub_pitch2 = sub_buff2.length / float(subsurf2.get_height())
                test_sub_pitch2 = subsurf2.get_pitch()
                self.assertEqual(sub_pitch2, test_sub_pitch2)

    def test_get_shifts(self):
        """
        Tests whether Surface.get_shifts returns proper
        RGBA shifts under various conditions.
        """
        depths = [8, 24, 32]
        alpha = 128
        off = None
        for bit_depth in depths:
            surface = pygame.Surface((32, 32), depth=bit_depth)
            surface.set_alpha(alpha)
            r1, g1, b1, a1 = surface.get_shifts()
            surface.set_alpha(off)
            r2, g2, b2, a2 = surface.get_shifts()
            self.assertEqual((r1, g1, b1, a1), (r2, g2, b2, a2))

    def test_get_size(self):
        sizes = ((1, 1), (119, 10), (1000, 1000), (1, 5000), (1221, 1), (99, 999))
        for width, height in sizes:
            surf = pygame.Surface((width, height))
            found_size = surf.get_size()
            self.assertEqual((width, height), found_size)

    def test_lock(self):
        surf = pygame.Surface((100, 100))
        surf.lock()
        self.assertTrue(surf.get_locked())
        surf = pygame.Surface((100, 100))
        surf.lock()
        surf.lock()
        surf.unlock()
        self.assertTrue(surf.get_locked())
        surf.unlock()
        surf.lock()
        surf.lock()
        self.assertTrue(surf.get_locked())
        surf.unlock()
        self.assertTrue(surf.get_locked())
        surf.unlock()
        self.assertFalse(surf.get_locked())
        surf = pygame.Surface((100, 100))
        surf.lock()
        surf.lock()
        self.assertTrue(surf.get_locked())
        surf.unlock()
        self.assertTrue(surf.get_locked())
        surf.unlock()
        self.assertFalse(surf.get_locked())

    def test_map_rgb(self):
        color = Color(0, 128, 255, 64)
        surf = pygame.Surface((5, 5), SRCALPHA, 32)
        c = surf.map_rgb(color)
        self.assertEqual(surf.unmap_rgb(c), color)
        self.assertEqual(surf.get_at((0, 0)), (0, 0, 0, 0))
        surf.fill(c)
        self.assertEqual(surf.get_at((0, 0)), color)
        surf.fill((0, 0, 0, 0))
        self.assertEqual(surf.get_at((0, 0)), (0, 0, 0, 0))
        surf.set_at((0, 0), c)
        self.assertEqual(surf.get_at((0, 0)), color)

    def test_mustlock(self):
        surf = pygame.Surface((1024, 1024))
        subsurf = surf.subsurface((0, 0, 1024, 1024))
        self.assertTrue(subsurf.mustlock())
        self.assertFalse(surf.mustlock())
        rects = ((0, 0, 512, 512), (0, 0, 256, 256), (0, 0, 128, 128))
        surf_stack = []
        surf_stack.append(surf)
        surf_stack.append(subsurf)
        for rect in rects:
            surf_stack.append(surf_stack[-1].subsurface(rect))
            self.assertTrue(surf_stack[-1].mustlock())
            self.assertTrue(surf_stack[-2].mustlock())

    def test_set_alpha_none(self):
        """surf.set_alpha(None) disables blending"""
        s = pygame.Surface((1, 1), SRCALPHA, 32)
        s.fill((0, 255, 0, 128))
        s.set_alpha(None)
        self.assertEqual(None, s.get_alpha())
        s2 = pygame.Surface((1, 1), SRCALPHA, 32)
        s2.fill((255, 0, 0, 255))
        s2.blit(s, (0, 0))
        self.assertEqual(s2.get_at((0, 0))[0], 0, 'the red component should be 0')

    def test_set_alpha_value(self):
        """surf.set_alpha(x), where x != None, enables blending"""
        s = pygame.Surface((1, 1), SRCALPHA, 32)
        s.fill((0, 255, 0, 128))
        s.set_alpha(255)
        s2 = pygame.Surface((1, 1), SRCALPHA, 32)
        s2.fill((255, 0, 0, 255))
        s2.blit(s, (0, 0))
        self.assertGreater(s2.get_at((0, 0))[0], 0, 'the red component should be above 0')

    def test_palette_colorkey(self):
        """test bug discovered by robertpfeiffer
        https://github.com/pygame/pygame/issues/721
        """
        surf = pygame.image.load(example_path(os.path.join('data', 'alien2.png')))
        key = surf.get_colorkey()
        self.assertEqual(surf.get_palette()[surf.map_rgb(key)], key)

    def test_palette_colorkey_set_px(self):
        surf = pygame.image.load(example_path(os.path.join('data', 'alien2.png')))
        key = surf.get_colorkey()
        surf.set_at((0, 0), key)
        self.assertEqual(surf.get_at((0, 0)), key)

    def test_palette_colorkey_fill(self):
        surf = pygame.image.load(example_path(os.path.join('data', 'alien2.png')))
        key = surf.get_colorkey()
        surf.fill(key)
        self.assertEqual(surf.get_at((0, 0)), key)

    def test_set_palette(self):
        palette = [pygame.Color(i, i, i) for i in range(256)]
        palette[10] = tuple(palette[10])
        palette[11] = tuple(palette[11])[0:3]
        surf = pygame.Surface((2, 2), 0, 8)
        surf.set_palette(palette)
        for i in range(256):
            self.assertEqual(surf.map_rgb(palette[i]), i, 'palette color %i' % (i,))
            c = palette[i]
            surf.fill(c)
            self.assertEqual(surf.get_at((0, 0)), c, 'palette color %i' % (i,))
        for i in range(10):
            palette[i] = pygame.Color(255 - i, 0, 0)
        surf.set_palette(palette[0:10])
        for i in range(256):
            self.assertEqual(surf.map_rgb(palette[i]), i, 'palette color %i' % (i,))
            c = palette[i]
            surf.fill(c)
            self.assertEqual(surf.get_at((0, 0)), c, 'palette color %i' % (i,))
        self.assertRaises(ValueError, surf.set_palette, [Color(1, 2, 3, 254)])
        self.assertRaises(ValueError, surf.set_palette, (1, 2, 3, 254))

    def test_set_palette__fail(self):
        palette = 256 * [(10, 20, 30)]
        surf = pygame.Surface((2, 2), 0, 32)
        self.assertRaises(pygame.error, surf.set_palette, palette)

    def test_set_palette__set_at(self):
        surf = pygame.Surface((2, 2), depth=8)
        palette = 256 * [(10, 20, 30)]
        palette[1] = (50, 40, 30)
        surf.set_palette(palette)
        surf.set_at((0, 0), (60, 50, 40))
        self.assertEqual(surf.get_at((0, 0)), (50, 40, 30, 255))
        self.assertEqual(surf.get_at((1, 0)), (10, 20, 30, 255))

    def test_set_palette_at(self):
        surf = pygame.Surface((2, 2), 0, 8)
        original = surf.get_palette_at(10)
        replacement = Color(1, 1, 1, 255)
        if replacement == original:
            replacement = Color(2, 2, 2, 255)
        surf.set_palette_at(10, replacement)
        self.assertEqual(surf.get_palette_at(10), replacement)
        next = tuple(original)
        surf.set_palette_at(10, next)
        self.assertEqual(surf.get_palette_at(10), next)
        next = tuple(original)[0:3]
        surf.set_palette_at(10, next)
        self.assertEqual(surf.get_palette_at(10), next)
        self.assertRaises(IndexError, surf.set_palette_at, 256, replacement)
        self.assertRaises(IndexError, surf.set_palette_at, -1, replacement)

    def test_subsurface(self):
        surf = pygame.Surface((16, 16))
        s = surf.subsurface(0, 0, 1, 1)
        s = surf.subsurface((0, 0, 1, 1))
        self.assertRaises(ValueError, surf.subsurface, (0, 0, 1, 1, 666))
        self.assertEqual(s.get_shifts(), surf.get_shifts())
        self.assertEqual(s.get_masks(), surf.get_masks())
        self.assertEqual(s.get_losses(), surf.get_losses())
        surf = pygame.Surface.__new__(pygame.Surface)
        self.assertRaises(pygame.error, surf.subsurface, (0, 0, 0, 0))

    def test_unlock(self):
        surf = pygame.Surface((100, 100))
        surf.lock()
        surf.unlock()
        self.assertFalse(surf.get_locked())
        surf = pygame.Surface((100, 100))
        surf.lock()
        surf.lock()
        surf.unlock()
        self.assertTrue(surf.get_locked())
        surf.unlock()
        self.assertFalse(surf.get_locked())
        surf = pygame.Surface((100, 100))
        surf.unlock()
        self.assertFalse(surf.get_locked())
        surf.unlock()
        self.assertFalse(surf.get_locked())
        surf = pygame.Surface((100, 100))
        surf.lock()
        surf.unlock()
        self.assertFalse(surf.get_locked())
        surf.lock()
        surf.unlock()
        self.assertFalse(surf.get_locked())

    def test_unmap_rgb(self):
        surf = pygame.Surface((2, 2), 0, 8)
        c = (1, 1, 1)
        i = 67
        surf.set_palette_at(i, c)
        unmapped_c = surf.unmap_rgb(i)
        self.assertEqual(unmapped_c, c)
        self.assertIsInstance(unmapped_c, pygame.Color)
        c = (128, 64, 12, 255)
        formats = [(0, 16), (0, 24), (0, 32), (SRCALPHA, 16), (SRCALPHA, 32)]
        for flags, bitsize in formats:
            surf = pygame.Surface((2, 2), flags, bitsize)
            unmapped_c = surf.unmap_rgb(surf.map_rgb(c))
            surf.fill(c)
            comparison_c = surf.get_at((0, 0))
            self.assertEqual(unmapped_c, comparison_c, '%s != %s, flags: %i, bitsize: %i' % (unmapped_c, comparison_c, flags, bitsize))
            self.assertIsInstance(unmapped_c, pygame.Color)

    def test_scroll(self):
        scrolls = [(8, 2, 3), (16, 2, 3), (24, 2, 3), (32, 2, 3), (32, -1, -3), (32, 0, 0), (32, 11, 0), (32, 0, 11), (32, -11, 0), (32, 0, -11), (32, -11, 2), (32, 2, -11)]
        for bitsize, dx, dy in scrolls:
            surf = pygame.Surface((10, 10), 0, bitsize)
            surf.fill((255, 0, 0))
            surf.fill((0, 255, 0), (2, 2, 2, 2))
            comp = surf.copy()
            comp.blit(surf, (dx, dy))
            surf.scroll(dx, dy)
            w, h = surf.get_size()
            for x in range(w):
                for y in range(h):
                    with self.subTest(x=x, y=y):
                        self.assertEqual(surf.get_at((x, y)), comp.get_at((x, y)), '%s != %s, bpp:, %i, x: %i, y: %i' % (surf.get_at((x, y)), comp.get_at((x, y)), bitsize, dx, dy))
        surf = pygame.Surface((20, 13), 0, 32)
        surf.fill((255, 0, 0))
        surf.fill((0, 255, 0), (7, 1, 6, 6))
        comp = surf.copy()
        clip = Rect(3, 1, 8, 14)
        surf.set_clip(clip)
        comp.set_clip(clip)
        comp.blit(surf, (clip.x + 2, clip.y + 3), surf.get_clip())
        surf.scroll(2, 3)
        w, h = surf.get_size()
        for x in range(w):
            for y in range(h):
                self.assertEqual(surf.get_at((x, y)), comp.get_at((x, y)))
        spot_color = (0, 255, 0, 128)
        surf = pygame.Surface((4, 4), pygame.SRCALPHA, 32)
        surf.fill((255, 0, 0, 255))
        surf.set_at((1, 1), spot_color)
        surf.scroll(dx=1)
        self.assertEqual(surf.get_at((2, 1)), spot_color)
        surf.scroll(dy=1)
        self.assertEqual(surf.get_at((2, 2)), spot_color)
        surf.scroll(dy=1, dx=1)
        self.assertEqual(surf.get_at((3, 3)), spot_color)
        surf.scroll(dx=-3, dy=-3)
        self.assertEqual(surf.get_at((0, 0)), spot_color)