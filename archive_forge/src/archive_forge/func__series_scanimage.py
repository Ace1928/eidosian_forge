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
def _series_scanimage(self) -> list[TiffPageSeries] | None:
    """Return image series in ScanImage file."""
    pages = self.pages._getlist(validate=False)
    page = self.pages.first
    dtype = page.dtype
    shape = None
    meta = self.scanimage_metadata
    if meta is None:
        framedata = {}
    else:
        framedata = meta.get('FrameData', {})
    if 'SI.hChannels.channelSave' in framedata:
        try:
            channels = framedata['SI.hChannels.channelSave']
            try:
                channels = len(channels)
            except TypeError:
                channels = 1
            slices = None
            try:
                frames = int(framedata['SI.hStackManager.framesPerSlice'])
            except Exception as exc:
                slices = 1
                if len(pages) % channels:
                    raise ValueError('unable to determine framesPerSlice') from exc
                frames = len(pages) // channels
            if slices is None:
                slices = max(len(pages) // (frames * channels), 1)
            shape = (slices, frames, channels) + page.shape
            axes = 'ZTC' + page.axes
        except Exception as exc:
            logger().warning(f'{self!r} ScanImage series raised {exc!r}')
    if shape is None:
        shape = (len(pages),) + page.shape
        axes = 'I' + page.axes
    return [TiffPageSeries(pages, shape, dtype, axes, kind='scanimage')]