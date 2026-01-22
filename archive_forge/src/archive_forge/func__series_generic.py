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
def _series_generic(self) -> list[TiffPageSeries] | None:
    """Return image series in file.

        A series is a sequence of TiffPages with the same hash.

        """
    pages = self.pages
    pages._clear(False)
    pages.useframes = False
    if pages.cache:
        pages._load()
    series = []
    keys = []
    seriesdict: dict[int, list[TiffPage | TiffFrame]] = {}

    def addpage(page: TiffPage | TiffFrame, /) -> None:
        if not page.shape:
            return
        key = page.hash
        if key in seriesdict:
            for p in seriesdict[key]:
                if p.offset == page.offset:
                    break
            else:
                seriesdict[key].append(page)
        else:
            keys.append(key)
            seriesdict[key] = [page]
    for page in pages:
        addpage(page)
        if page.subifds is not None:
            for i, offset in enumerate(page.subifds):
                if offset < 8:
                    continue
                try:
                    self._fh.seek(offset)
                    subifd = TiffPage(self, (page.index, i))
                except Exception as exc:
                    logger().warning(f'{self!r} generic series raised {exc!r}')
                else:
                    addpage(subifd)
    for key in keys:
        pagelist = seriesdict[key]
        page = pagelist[0]
        shape = (len(pagelist),) + page.shape
        axes = 'I' + page.axes
        if 'S' not in axes:
            shape += (1,)
            axes += 'S'
        series.append(TiffPageSeries(pagelist, shape, page.dtype, axes, kind='generic'))
    self.is_uniform = len(series) == 1
    if not self.is_agilent:
        pyramidize_series(series)
    return series