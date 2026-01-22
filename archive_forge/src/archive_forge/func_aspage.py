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
def aspage(self):
    """Return TiffPage from file."""
    self.parent.filehandle.seek(self.offset)
    return TiffPage(self.parent, index=self.index, keyframe=None)