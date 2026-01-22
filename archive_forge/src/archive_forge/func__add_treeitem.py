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
def _add_treeitem(self, memory, name, path=None):
    buff = JM_BufferFromBytes(memory)
    sub = mupdf.fz_new_tree_archive(mupdf.FzTree())
    mupdf.fz_tree_archive_add_buffer(sub, name, buff)
    mupdf.fz_mount_multi_archive(self.this, sub, path)