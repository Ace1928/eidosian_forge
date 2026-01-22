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
def JM_new_bbox_device(rc, inc_layers):
    assert isinstance(rc, list)
    return JM_new_bbox_device_Device(rc, inc_layers)