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
def test_premul_surf(src_col, dst_col, src_size=(16, 16), dst_size=(16, 16), src_bit_depth=32, dst_bit_depth=32, src_has_alpha=True, dst_has_alpha=True):
    if src_bit_depth == 8:
        src = pygame.Surface(src_size, 0, src_bit_depth)
        palette = [src_col, dst_col]
        src.set_palette(palette)
        src.fill(palette[0])
    elif src_has_alpha:
        src = pygame.Surface(src_size, SRCALPHA, src_bit_depth)
        src.fill(src_col)
    else:
        src = pygame.Surface(src_size, 0, src_bit_depth)
        src.fill(src_col)
    if dst_bit_depth == 8:
        dst = pygame.Surface(dst_size, 0, dst_bit_depth)
        palette = [src_col, dst_col]
        dst.set_palette(palette)
        dst.fill(palette[1])
    elif dst_has_alpha:
        dst = pygame.Surface(dst_size, SRCALPHA, dst_bit_depth)
        dst.fill(dst_col)
    else:
        dst = pygame.Surface(dst_size, 0, dst_bit_depth)
        dst.fill(dst_col)
    dst.blit(src, (0, 0), special_flags=BLEND_PREMULTIPLIED)
    actual_col = dst.get_at((int(float(src_size[0] / 2.0)), int(float(src_size[0] / 2.0))))
    if src_col.a == 0:
        expected_col = dst_col
    elif src_col.a == 255:
        expected_col = src_col
    else:
        expected_col = pygame.Color(src_col.r + dst_col.r - ((dst_col.r + 1) * src_col.a >> 8), src_col.g + dst_col.g - ((dst_col.g + 1) * src_col.a >> 8), src_col.b + dst_col.b - ((dst_col.b + 1) * src_col.a >> 8), src_col.a + dst_col.a - ((dst_col.a + 1) * src_col.a >> 8))
    if not dst_has_alpha:
        expected_col.a = 255
    return (expected_col, actual_col)