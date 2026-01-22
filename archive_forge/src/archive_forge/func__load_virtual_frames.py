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
def _load_virtual_frames(self) -> None:
    """Calculate virtual TiffFrames."""
    assert self.parent is not None
    pages = self.pages
    try:
        if len(pages) > 1:
            raise ValueError('pages already loaded')
        page = cast(TiffPage, pages[0])
        if not page.is_contiguous:
            raise ValueError('data not contiguous')
        self._seek(4)
        delta = cast(int, pages[2]) - cast(int, pages[1])
        if cast(int, pages[3]) - cast(int, pages[2]) != delta or cast(int, pages[4]) - cast(int, pages[3]) != delta:
            raise ValueError('page offsets not equidistant')
        page1 = self._getitem(1, validate=page.hash)
        offsetoffset = page1.dataoffsets[0] - page1.offset
        if offsetoffset < 0 or offsetoffset > delta:
            raise ValueError('page offsets not equidistant')
        pages = [page, page1]
        filesize = self.parent.filehandle.size - delta
        for index, offset in enumerate(range(page1.offset + delta, filesize, delta)):
            index += 2
            d = index * delta
            dataoffsets = tuple((i + d for i in page.dataoffsets))
            offset_or_none = offset if offset < 2 ** 31 - 1 else None
            pages.append(TiffFrame(page.parent, index=index if self._index is None else self._index + (index,), offset=offset_or_none, dataoffsets=dataoffsets, databytecounts=page.databytecounts, keyframe=page))
        self.pages = pages
        self._cache = True
        self._cached = True
        self._indexed = True
    except Exception as exc:
        if self.parent.filehandle.size >= 2147483648:
            logger().warning(f'{self!r} <_load_virtual_frames> raised {exc!r}')