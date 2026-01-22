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
def _series_imagej(self) -> list[TiffPageSeries] | None:
    """Return image series in ImageJ file."""
    meta = self.imagej_metadata
    if meta is None:
        return None
    pages = self.pages
    pages.useframes = True
    pages.set_keyframe(0)
    page = self.pages.first
    order = meta.get('order', 'czt').lower()
    frames = meta.get('frames', 1)
    slices = meta.get('slices', 1)
    channels = meta.get('channels', 1)
    images = meta.get('images', 1)
    if images < 1 or frames < 1 or slices < 1 or (channels < 1):
        logger().warning(f'{self!r} ImageJ series metadata invalid or corrupted file')
        return None
    if channels == 1:
        images = frames * slices
    elif page.shaped[0] > 1 and page.shaped[0] == channels:
        images = frames * slices
    elif images == frames * slices and page.shaped[4] == channels:
        channels = 1
    else:
        images = frames * slices * channels
    if images == 1 and pages.is_multipage:
        images = len(pages)
    nbytes = images * page.nbytes
    if not page.is_final:
        isvirtual = False
    elif page.dataoffsets[0] + nbytes > self.filehandle.size:
        logger().error(f'{self!r} ImageJ series metadata invalid or corrupted file')
        return None
    elif images <= 1:
        isvirtual = True
    elif pages.is_multipage and page.dataoffsets[0] + nbytes > pages[1].offset:
        isvirtual = False
    else:
        isvirtual = True
    page_list: list[TiffPage | TiffFrame]
    if isvirtual:
        page_list = [page]
    else:
        page_list = pages[:]
    shape: tuple[int, ...]
    axes: str
    if order in {'czt', 'default'}:
        axes = 'TZC'
        shape = (frames, slices, channels)
    elif order == 'ctz':
        axes = 'ZTC'
        shape = (slices, frames, channels)
    elif order == 'zct':
        axes = 'TCZ'
        shape = (frames, channels, slices)
    elif order == 'ztc':
        axes = 'CTZ'
        shape = (channels, frames, slices)
    elif order == 'tcz':
        axes = 'ZCT'
        shape = (slices, channels, frames)
    elif order == 'tzc':
        axes = 'CZT'
        shape = (channels, slices, frames)
    else:
        axes = 'TZC'
        shape = (frames, slices, channels)
        logger().warning(f'{self!r} ImageJ series of unknown order {order!r}')
    remain = images // product(shape)
    if remain > 1:
        logger().debug(f'{self!r} ImageJ series contains unidentified dimension')
        shape = (remain,) + shape
        axes = 'I' + axes
    if page.shaped[0] > 1:
        assert axes[-1] == 'C'
        shape = shape[:-1] + page.shape
        axes += page.axes[1:]
    else:
        shape += page.shape
        axes += page.axes
    if 'S' not in axes:
        shape += (1,)
        axes += 'S'
    truncated = isvirtual and (not pages.is_multipage) and (page.nbytes != nbytes)
    self.is_uniform = True
    return [TiffPageSeries(page_list, shape, page.dtype, axes, kind='imagej', truncated=truncated)]