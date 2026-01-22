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
def FILETYPE():

    class FILETYPE(enum.IntFlag):
        UNDEFINED = 0
        REDUCEDIMAGE = 1
        PAGE = 2
        MASK = 4
    return FILETYPE