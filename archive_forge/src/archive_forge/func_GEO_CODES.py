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
def GEO_CODES():
    try:
        from .tifffile_geodb import GEO_CODES
    except (ImportError, ValueError):
        try:
            from tifffile_geodb import GEO_CODES
        except (ImportError, ValueError):
            GEO_CODES = {}
    return GEO_CODES