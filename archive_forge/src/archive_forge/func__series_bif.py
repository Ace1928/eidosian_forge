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
def _series_bif(self) -> list[TiffPageSeries] | None:
    """Return image series in Ventana/Roche BIF file."""
    series = []
    baseline = None
    self.pages.cache = True
    self.pages.useframes = False
    self.pages.set_keyframe(0)
    self.pages._load()
    for page in self.pages:
        page = cast(TiffPage, page)
        if page.description[:5] == 'Label':
            series.append(TiffPageSeries([page], page.shape, page.dtype, page.axes, name='Label', kind='bif'))
        elif page.description == 'Thumbnail' or page.description[:11] == 'Probability':
            series.append(TiffPageSeries([page], page.shape, page.dtype, page.axes, name='Thumbnail', kind='bif'))
        elif 'level' not in page.description:
            series.append(TiffPageSeries([page], page.shape, page.dtype, page.axes, name='Unknown', kind='bif'))
        elif baseline is None:
            baseline = TiffPageSeries([page], page.shape, page.dtype, page.axes, name='Baseline', kind='bif')
            series.insert(0, baseline)
        else:
            baseline.levels.append(TiffPageSeries([page], page.shape, page.dtype, page.axes, name='Resolution', kind='bif'))
    logger().warning(f'{self!r} BIF series tiles are not stiched')
    self.is_uniform = False
    return series