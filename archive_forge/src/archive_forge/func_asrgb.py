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
def asrgb(self, *args, **kwargs):
    """Read image data from file and return RGB image as numpy array."""
    kwargs['validate'] = False
    return TiffPage.asrgb(self, *args, **kwargs)