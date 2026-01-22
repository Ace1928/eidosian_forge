import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def _NEWBUF_export_check(self):
    freq, fmt, channels = mixer.get_init()
    ndim = 1 if channels == 1 else 2
    itemsize = abs(fmt) // 8
    formats = {8: 'B', -8: 'b', 16: '=H', -16: '=h', 32: '=I', -32: '=i', 64: '=Q', -64: '=q'}
    format = formats[fmt]
    from pygame.tests.test_utils import buftools
    Exporter = buftools.Exporter
    Importer = buftools.Importer
    is_lil_endian = pygame.get_sdl_byteorder() == pygame.LIL_ENDIAN
    fsys, frev = ('<', '>') if is_lil_endian else ('>', '<')
    shape = (10, channels)[:ndim]
    strides = (channels * itemsize, itemsize)[2 - ndim:]
    exp = Exporter(shape, format=frev + 'i')
    snd = mixer.Sound(array=exp)
    buflen = len(exp) * itemsize * channels
    imp = Importer(snd, buftools.PyBUF_SIMPLE)
    self.assertEqual(imp.ndim, 0)
    self.assertTrue(imp.format is None)
    self.assertEqual(imp.len, buflen)
    self.assertEqual(imp.itemsize, itemsize)
    self.assertTrue(imp.shape is None)
    self.assertTrue(imp.strides is None)
    self.assertTrue(imp.suboffsets is None)
    self.assertFalse(imp.readonly)
    self.assertEqual(imp.buf, snd._samples_address)
    imp = Importer(snd, buftools.PyBUF_WRITABLE)
    self.assertEqual(imp.ndim, 0)
    self.assertTrue(imp.format is None)
    self.assertEqual(imp.len, buflen)
    self.assertEqual(imp.itemsize, itemsize)
    self.assertTrue(imp.shape is None)
    self.assertTrue(imp.strides is None)
    self.assertTrue(imp.suboffsets is None)
    self.assertFalse(imp.readonly)
    self.assertEqual(imp.buf, snd._samples_address)
    imp = Importer(snd, buftools.PyBUF_FORMAT)
    self.assertEqual(imp.ndim, 0)
    self.assertEqual(imp.format, format)
    self.assertEqual(imp.len, buflen)
    self.assertEqual(imp.itemsize, itemsize)
    self.assertTrue(imp.shape is None)
    self.assertTrue(imp.strides is None)
    self.assertTrue(imp.suboffsets is None)
    self.assertFalse(imp.readonly)
    self.assertEqual(imp.buf, snd._samples_address)
    imp = Importer(snd, buftools.PyBUF_ND)
    self.assertEqual(imp.ndim, ndim)
    self.assertTrue(imp.format is None)
    self.assertEqual(imp.len, buflen)
    self.assertEqual(imp.itemsize, itemsize)
    self.assertEqual(imp.shape, shape)
    self.assertTrue(imp.strides is None)
    self.assertTrue(imp.suboffsets is None)
    self.assertFalse(imp.readonly)
    self.assertEqual(imp.buf, snd._samples_address)
    imp = Importer(snd, buftools.PyBUF_STRIDES)
    self.assertEqual(imp.ndim, ndim)
    self.assertTrue(imp.format is None)
    self.assertEqual(imp.len, buflen)
    self.assertEqual(imp.itemsize, itemsize)
    self.assertEqual(imp.shape, shape)
    self.assertEqual(imp.strides, strides)
    self.assertTrue(imp.suboffsets is None)
    self.assertFalse(imp.readonly)
    self.assertEqual(imp.buf, snd._samples_address)
    imp = Importer(snd, buftools.PyBUF_FULL_RO)
    self.assertEqual(imp.ndim, ndim)
    self.assertEqual(imp.format, format)
    self.assertEqual(imp.len, buflen)
    self.assertEqual(imp.itemsize, 2)
    self.assertEqual(imp.shape, shape)
    self.assertEqual(imp.strides, strides)
    self.assertTrue(imp.suboffsets is None)
    self.assertFalse(imp.readonly)
    self.assertEqual(imp.buf, snd._samples_address)
    imp = Importer(snd, buftools.PyBUF_FULL_RO)
    self.assertEqual(imp.ndim, ndim)
    self.assertEqual(imp.format, format)
    self.assertEqual(imp.len, buflen)
    self.assertEqual(imp.itemsize, itemsize)
    self.assertEqual(imp.shape, exp.shape)
    self.assertEqual(imp.strides, strides)
    self.assertTrue(imp.suboffsets is None)
    self.assertFalse(imp.readonly)
    self.assertEqual(imp.buf, snd._samples_address)
    imp = Importer(snd, buftools.PyBUF_C_CONTIGUOUS)
    self.assertEqual(imp.ndim, ndim)
    self.assertTrue(imp.format is None)
    self.assertEqual(imp.strides, strides)
    imp = Importer(snd, buftools.PyBUF_ANY_CONTIGUOUS)
    self.assertEqual(imp.ndim, ndim)
    self.assertTrue(imp.format is None)
    self.assertEqual(imp.strides, strides)
    if ndim == 1:
        imp = Importer(snd, buftools.PyBUF_F_CONTIGUOUS)
        self.assertEqual(imp.ndim, 1)
        self.assertTrue(imp.format is None)
        self.assertEqual(imp.strides, strides)
    else:
        self.assertRaises(BufferError, Importer, snd, buftools.PyBUF_F_CONTIGUOUS)