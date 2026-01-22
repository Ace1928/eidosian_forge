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
def MM_DIMENSIONS():
    return {'X': 'X', 'Y': 'Y', 'Z': 'Z', 'T': 'T', 'CH': 'C', 'WAVELENGTH': 'C', 'TIME': 'T', 'XY': 'R', 'EVENT': 'V', 'EXPOSURE': 'L'}