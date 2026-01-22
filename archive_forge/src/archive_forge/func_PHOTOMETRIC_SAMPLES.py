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
def PHOTOMETRIC_SAMPLES(self) -> dict[int, int]:
    """Map :py:class:`PHOTOMETRIC` to number of photometric samples."""
    return {0: 1, 1: 1, 2: 3, 3: 1, 4: 1, 5: 4, 6: 3, 8: 3, 9: 3, 10: 3, 32803: 1, 32844: 1, 32845: 3, 34892: 3, 51177: 1, 52527: 1}