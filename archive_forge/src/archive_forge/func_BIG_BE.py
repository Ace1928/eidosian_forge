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
def BIG_BE(self) -> TiffFormat:
    """64-bit big-endian TIFF format."""
    return TiffFormat(version=43, byteorder='>', offsetsize=8, offsetformat='>Q', tagnosize=8, tagnoformat='>Q', tagsize=20, tagformat1='>HH', tagformat2='>Q8s', tagoffsetthreshold=8)