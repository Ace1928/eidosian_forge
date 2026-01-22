from __future__ import annotations
import io
import itertools
import logging
import math
import os
import struct
import warnings
from collections.abc import MutableMapping
from fractions import Fraction
from numbers import Number, Rational
from . import ExifTags, Image, ImageFile, ImageOps, ImagePalette, TiffTags
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from .TiffTags import TYPES
def _load_libtiff(self):
    """Overload method triggered when we detect a compressed tiff
        Calls out to libtiff"""
    Image.Image.load(self)
    self.load_prepare()
    if not len(self.tile) == 1:
        msg = 'Not exactly one tile'
        raise OSError(msg)
    extents = self.tile[0][1]
    args = list(self.tile[0][3])
    try:
        fp = hasattr(self.fp, 'fileno') and self.fp.fileno()
        if hasattr(self.fp, 'flush'):
            self.fp.flush()
    except OSError:
        fp = False
    if fp:
        args[2] = fp
    decoder = Image._getdecoder(self.mode, 'libtiff', tuple(args), self.decoderconfig)
    try:
        decoder.setimage(self.im, extents)
    except ValueError as e:
        msg = "Couldn't set the image"
        raise OSError(msg) from e
    close_self_fp = self._exclusive_fp and (not self.is_animated)
    if hasattr(self.fp, 'getvalue'):
        logger.debug('have getvalue. just sending in a string from getvalue')
        n, err = decoder.decode(self.fp.getvalue())
    elif fp:
        logger.debug('have fileno, calling fileno version of the decoder.')
        if not close_self_fp:
            self.fp.seek(0)
        n, err = decoder.decode(b'fpfp')
    else:
        logger.debug("don't have fileno or getvalue. just reading")
        self.fp.seek(0)
        n, err = decoder.decode(self.fp.read())
    self.tile = []
    self.readonly = 0
    self.load_end()
    if close_self_fp:
        self.fp.close()
        self.fp = None
    if err < 0:
        raise OSError(err)
    return Image.Image.load(self)