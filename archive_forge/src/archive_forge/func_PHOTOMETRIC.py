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
def PHOTOMETRIC():

    class PHOTOMETRIC(enum.IntEnum):
        MINISWHITE = 0
        MINISBLACK = 1
        RGB = 2
        PALETTE = 3
        MASK = 4
        SEPARATED = 5
        YCBCR = 6
        CIELAB = 8
        ICCLAB = 9
        ITULAB = 10
        CFA = 32803
        LOGL = 32844
        LOGLUV = 32845
        LINEAR_RAW = 34892
    return PHOTOMETRIC