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
def TAG_FILTERED(self) -> frozenset[int]:
    return frozenset((256, 257, 258, 259, 262, 266, 273, 277, 278, 279, 284, 317, 322, 323, 324, 325, 330, 338, 339, 400, 32997, 32998, 34665, 34853, 40965))