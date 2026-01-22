from __future__ import annotations
import binascii
import collections
import datetime
import enum
import glob
import io
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import numpy
from typing import TYPE_CHECKING, BinaryIO, cast, final, overload
def _series_mmstack(self) -> list[TiffPageSeries] | None:
    """Return series in Micro-Manager stack file(s)."""
    settings = self.micromanager_metadata
    if settings is None or 'Summary' not in settings or 'IndexMap' not in settings:
        return None
    pages: list[TiffPage | TiffFrame | None]
    page_count: int
    summary = settings['Summary']
    indexmap = settings['IndexMap']
    indexmap = indexmap[indexmap[:, 4].argsort()]
    if 'MicroManagerVersion' not in summary or 'Frames' not in summary:
        return None
    indexmap_shape = (numpy.max(indexmap[:, :4], axis=0) + 1).tolist()
    indexmap_index = {'C': 0, 'Z': 1, 'T': 2, 'R': 3}
    axes = 'TR' if summary.get('TimeFirst', True) else 'RT'
    axes += 'ZC' if summary.get('SlicesFirst', True) else 'CZ'
    keys = {'C': 'Channels', 'Z': 'Slices', 'R': 'Positions', 'T': 'Frames'}
    shape = tuple((max(indexmap_shape[indexmap_index[ax]], int(summary.get(keys[ax], 1))) for ax in axes))
    size = product(shape)
    indexmap_order = tuple((indexmap_index[ax] for ax in axes))

    def add_file(tif: TiffFile, indexmap: NDArray[Any]) -> int:
        page_count = 0
        offsets = indexmap[:, 4].tolist()
        indices = numpy.ravel_multi_index(indexmap[:, indexmap_order].T, shape).tolist()
        keyframe = tif.pages.first
        filesize = tif.filehandle.size - keyframe.databytecounts[0] - 162
        index: int
        offset: int
        for index, offset in zip(indices, offsets):
            if offset == keyframe.offset:
                pages[index] = keyframe
                page_count += 1
                continue
            if 0 < offset <= filesize:
                dataoffsets = (offset + 162,)
                databytecounts = keyframe.databytecounts
                page_count += 1
            else:
                dataoffsets = databytecounts = (0,)
                offset = 0
            pages[index] = TiffFrame(tif, index=index, offset=offset, dataoffsets=dataoffsets, databytecounts=databytecounts, keyframe=keyframe)
        return page_count
    multifile = size > indexmap.shape[0]
    if multifile:
        if not self.filehandle.is_file:
            logger().warning(f'{self!r} MMStack multi-file series cannot be read from {self.filehandle._fh!r}')
            multifile = False
        elif '_MMStack' not in self.filename:
            logger().warning(f'{self!r} MMStack file name is invalid')
            multifile = False
        elif 'Prefix' in summary:
            prefix = summary['Prefix']
            if not self.filename.startswith(prefix):
                logger().warning(f'{self!r} MMStack file name is invalid')
                multifile = False
        else:
            prefix = self.filename.split('_MMStack')[0]
    if multifile:
        pattern = os.path.join(self.filehandle.dirname, prefix + '_MMStack*.tif')
        filenames = glob.glob(pattern)
        if len(filenames) == 1:
            multifile = False
        else:
            pages = [None] * size
            page_count = add_file(self, indexmap)
            for fname in filenames:
                if self.filename == os.path.split(fname)[-1]:
                    continue
                with TiffFile(fname) as tif:
                    indexmap = read_micromanager_metadata(tif.filehandle, {'IndexMap'})['IndexMap']
                    indexmap = indexmap[indexmap[:, 4].argsort()]
                    page_count += add_file(tif, indexmap)
    if multifile:
        pass
    elif size > indexmap.shape[0]:
        old_shape = shape
        min_index = numpy.min(indexmap[:, :4], axis=0)
        max_index = numpy.max(indexmap[:, :4], axis=0)
        indexmap = indexmap.copy()
        indexmap[:, :4] -= min_index
        shape = tuple((j - i + 1 for i, j in zip(min_index.tolist(), max_index.tolist())))
        shape = tuple((shape[i] for i in indexmap_order))
        size = product(shape)
        pages = [None] * size
        page_count = add_file(self, indexmap)
        logger().warning(f'{self!r} MMStack series is missing files. Returning subset {shape!r} of {old_shape!r}')
    else:
        pages = [None] * size
        page_count = add_file(self, indexmap)
    if page_count != size:
        logger().warning(f'{self!r} MMStack is missing {size - page_count} pages. Missing data are zeroed')
    keyframe = self.pages.first
    return [TiffPageSeries(pages, shape=shape + keyframe.shape, dtype=keyframe.dtype, axes=axes + keyframe.axes, parent=self, kind='mmstack', multifile=multifile, squeeze=True)]