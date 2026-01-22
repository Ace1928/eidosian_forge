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
class CZ_LSM_LUTTYPE(enum.IntEnum):
    NORMAL = 0
    ORIGINAL = 1
    RAMP = 2
    POLYLINE = 3
    SPLINE = 4
    GAMMA = 5