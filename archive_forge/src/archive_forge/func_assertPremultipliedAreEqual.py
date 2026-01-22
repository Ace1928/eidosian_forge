import array
import binascii
import io
import os
import tempfile
import unittest
import glob
import pathlib
from pygame.tests.test_utils import example_path, png, tostring
import pygame, pygame.image, pygame.pkgdata
def assertPremultipliedAreEqual(self, string1, string2, source_string):
    self.assertEqual(len(string1), len(string2))
    block_size = 20
    if string1 != string2:
        for block_start in range(0, len(string1), block_size):
            block_end = min(block_start + block_size, len(string1))
            block1 = string1[block_start:block_end]
            block2 = string2[block_start:block_end]
            if block1 != block2:
                source_block = source_string[block_start:block_end]
                msg = 'string difference in %d to %d of %d:\n%s\n%s\nsource:\n%s' % (block_start, block_end, len(string1), binascii.hexlify(block1), binascii.hexlify(block2), binascii.hexlify(source_block))
                self.fail(msg)