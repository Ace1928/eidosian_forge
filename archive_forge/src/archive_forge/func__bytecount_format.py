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
def _bytecount_format(self, bytecounts: Sequence[int], compression: int, /) -> str:
    """Return small bytecount format."""
    if len(bytecounts) == 1:
        return self.tiff.offsetformat[1]
    bytecount = bytecounts[0]
    if compression > 1:
        bytecount = bytecount * 10
    if bytecount < 2 ** 16:
        return 'H'
    if bytecount < 2 ** 32:
        return 'I'
    return self.tiff.offsetformat[1]