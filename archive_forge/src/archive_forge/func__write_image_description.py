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
def _write_image_description(self):
    """Write meta data to ImageDescription tag."""
    if not self._datashape or self._datashape[0] == 1 or self._descriptionoffset <= 0:
        return
    colormapped = self._colormap is not None
    if self._imagej:
        isrgb = self._shape[-1] in (3, 4)
        description = imagej_description(self._datashape, isrgb, colormapped, **self._metadata)
    else:
        description = json_description(self._datashape, **self._metadata)
    description = description.encode('utf-8')
    description = description[:self._descriptionlen - 1]
    pos = self._fh.tell()
    self._fh.seek(self._descriptionoffset)
    self._fh.write(description)
    self._fh.seek(self._descriptionlenoffset)
    self._fh.write(struct.pack(self._byteorder + self._offsetformat, len(description) + 1))
    self._fh.seek(pos)
    self._descriptionoffset = 0
    self._descriptionlenoffset = 0
    self._descriptionlen = 0