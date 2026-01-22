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
def _nih_series(self):
    """Return image series in NIH file."""
    self.pages.useframes = True
    self.pages.keyframe = 0
    self.pages.load()
    page0 = self.pages[0]
    if len(self.pages) == 1:
        shape = page0.shape
        axes = page0.axes
    else:
        shape = (len(self.pages),) + page0.shape
        axes = 'I' + page0.axes
    return [TiffPageSeries(self.pages, shape, page0.dtype, axes, stype='NIH')]