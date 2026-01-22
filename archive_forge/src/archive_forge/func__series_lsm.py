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
def _series_lsm(self) -> list[TiffPageSeries] | None:
    """Return main and thumbnail series in LSM file."""
    lsmi = self.lsm_metadata
    if lsmi is None:
        return None
    axes = TIFF.CZ_LSMINFO_SCANTYPE[lsmi['ScanType']]
    if self.pages.first.photometric == 2:
        axes = axes.replace('C', '').replace('XY', 'XYC')
    if lsmi.get('DimensionP', 0) > 0:
        axes += 'P'
    if lsmi.get('DimensionM', 0) > 0:
        axes += 'M'
    axes = axes[::-1]
    shape = tuple((int(lsmi[TIFF.CZ_LSMINFO_DIMENSIONS[i]]) for i in axes))
    name = lsmi.get('Name', '')
    pages = self.pages._getlist(slice(0, None, 2), validate=False)
    dtype = pages[0].dtype
    series = [TiffPageSeries(pages, shape, dtype, axes, name=name, kind='lsm')]
    page = cast(TiffPage, self.pages[1])
    if page.is_reduced:
        pages = self.pages._getlist(slice(1, None, 2), validate=False)
        dtype = page.dtype
        cp = 1
        i = 0
        while cp < len(pages) and i < len(shape) - 2:
            cp *= shape[i]
            i += 1
        shape = shape[:i] + page.shape
        axes = axes[:i] + page.axes
        series.append(TiffPageSeries(pages, shape, dtype, axes, name=name, kind='lsm'))
    self.is_uniform = False
    return series