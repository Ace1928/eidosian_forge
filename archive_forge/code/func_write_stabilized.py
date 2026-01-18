import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
@staticmethod
def write_stabilized(writer, contentfn, rectfn, user_css=None, em=12, positionfn=None, pagefn=None, archive=None, add_header_ids=True):
    positions = list()
    content = None
    while 1:
        content_prev = content
        content = contentfn(positions)
        stable = False
        if content == content_prev:
            stable = True
        content2 = content
        story = Story(content2, user_css, em, archive)
        if add_header_ids:
            story.add_header_ids()
        positions = list()

        def positionfn2(position):
            positions.append(position)
            if stable and positionfn:
                positionfn(position)
        story.write(writer if stable else None, rectfn, positionfn2, pagefn)
        if stable:
            break