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
def append_series(series, pages, axes, shape, reshape, name, truncated):
    page = pages[0]
    if not axes:
        shape = page.shape
        axes = page.axes
        if len(pages) > 1:
            shape = (len(pages),) + shape
            axes = 'Q' + axes
    size = product(shape)
    resize = product(reshape)
    if page.is_contiguous and resize > size and (resize % size == 0):
        if truncated is None:
            truncated = True
        axes = 'Q' + axes
        shape = (resize // size,) + shape
    try:
        axes = reshape_axes(axes, shape, reshape)
        shape = reshape
    except ValueError as e:
        warnings.warn(str(e))
    series.append(TiffPageSeries(pages, shape, page.dtype, axes, name=name, stype='Shaped', truncated=truncated))