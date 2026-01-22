from __future__ import division, print_function
import sys
import os
import io
import re
import glob
import math
import zlib
import time
import json
import enum
import struct
import pathlib
import warnings
import binascii
import tempfile
import datetime
import threading
import collections
import multiprocessing
import concurrent.futures
import numpy
def _imagej_series(self):
    """Return image series in ImageJ file."""
    self.pages.useframes = True
    self.pages.keyframe = 0
    ij = self.imagej_metadata
    pages = self.pages
    page = pages[0]

    def is_hyperstack():
        if not page.is_final:
            return False
        images = ij.get('images', 0)
        if images <= 1:
            return False
        offset, count = page.is_contiguous
        if count != product(page.shape) * page.bitspersample // 8 or offset + count * images > self.filehandle.size:
            raise ValueError()
        if len(pages) > 1 and offset + count * images > pages[1].offset:
            return False
        return True
    try:
        hyperstack = is_hyperstack()
    except ValueError:
        warnings.warn('invalid ImageJ metadata or corrupted file')
        return
    if hyperstack:
        pages = [page]
    else:
        self.pages.load()
    shape = []
    axes = []
    if 'frames' in ij:
        shape.append(ij['frames'])
        axes.append('T')
    if 'slices' in ij:
        shape.append(ij['slices'])
        axes.append('Z')
    if 'channels' in ij and (not (page.photometric == 2 and (not ij.get('hyperstack', False)))):
        shape.append(ij['channels'])
        axes.append('C')
    remain = ij.get('images', len(pages)) // (product(shape) if shape else 1)
    if remain > 1:
        shape.append(remain)
        axes.append('I')
    if page.axes[0] == 'I':
        shape.extend(page.shape[1:])
        axes.extend(page.axes[1:])
    elif page.axes[:2] == 'SI':
        shape = page.shape[0:1] + tuple(shape) + page.shape[2:]
        axes = list(page.axes[0]) + axes + list(page.axes[2:])
    else:
        shape.extend(page.shape)
        axes.extend(page.axes)
    truncated = hyperstack and len(self.pages) == 1 and (page.is_contiguous[1] != product(shape) * page.bitspersample // 8)
    return [TiffPageSeries(pages, shape, page.dtype, axes, stype='ImageJ', truncated=truncated)]