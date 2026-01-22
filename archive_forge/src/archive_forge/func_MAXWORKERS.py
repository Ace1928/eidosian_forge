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
@cached_property
def MAXWORKERS(self) -> int:
    """Default maximum number of threads for de/compressing segments.

        The value of the ``TIFFFILE_NUM_THREADS`` environment variable if set,
        else half the CPU cores up to 32.

        """
    if 'TIFFFILE_NUM_THREADS' in os.environ:
        return max(1, int(os.environ['TIFFFILE_NUM_THREADS']))
    cpu_count: int | None
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count()
    if cpu_count is None:
        return 1
    return min(32, max(1, cpu_count // 2))