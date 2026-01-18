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
def update_kwargs(kwargs, **keyvalues):
    """Update dict with keys and values if keys do not already exist.

    >>> kwargs = {'one': 1, }
    >>> update_kwargs(kwargs, one=None, two=2)
    >>> kwargs == {'one': 1, 'two': 2}
    True

    """
    for key, value in keyvalues.items():
        if key not in kwargs:
            kwargs[key] = value