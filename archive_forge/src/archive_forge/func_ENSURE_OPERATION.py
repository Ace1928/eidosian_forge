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
def ENSURE_OPERATION(pdf):
    if not JM_have_operation(pdf):
        raise Exception('No journalling operation started')