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
def TAG_ENUM():
    return {255: TIFF.OFILETYPE, 259: TIFF.COMPRESSION, 262: TIFF.PHOTOMETRIC, 263: TIFF.THRESHHOLD, 266: TIFF.FILLORDER, 274: TIFF.ORIENTATION, 284: TIFF.PLANARCONFIG, 290: TIFF.GRAYRESPONSEUNIT, 296: TIFF.RESUNIT, 300: TIFF.COLORRESPONSEUNIT, 317: TIFF.PREDICTOR, 338: TIFF.EXTRASAMPLE, 339: TIFF.SAMPLEFORMAT}