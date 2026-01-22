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
def _series_uniform(self) -> list[TiffPageSeries] | None:
    """Return all images in file as single series."""
    self.pages.useframes = True
    self.pages.set_keyframe(0)
    page = self.pages.first
    validate = not (page.is_scanimage or page.is_nih)
    pages = self.pages._getlist(validate=validate)
    if len(pages) == 1:
        shape = page.shape
        axes = page.axes
    else:
        shape = (len(pages),) + page.shape
        axes = 'I' + page.axes
    dtype = page.dtype
    return [TiffPageSeries(pages, shape, dtype, axes, kind='uniform')]