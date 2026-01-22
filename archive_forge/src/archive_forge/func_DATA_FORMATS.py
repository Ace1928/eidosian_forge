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
def DATA_FORMATS():
    return {1: '1B', 2: '1s', 3: '1H', 4: '1I', 5: '2I', 6: '1b', 7: '1B', 8: '1h', 9: '1i', 10: '2i', 11: '1f', 12: '1d', 13: '1I', 16: '1Q', 17: '1q', 18: '1Q'}