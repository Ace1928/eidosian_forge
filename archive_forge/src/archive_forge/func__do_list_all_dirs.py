import contextlib
import datetime
import os
import pathlib
import posixpath
import sys
import tempfile
import textwrap
import threading
import time
from shutil import copytree
from urllib.parse import quote
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv
import pyarrow.feather
import pyarrow.fs as fs
import pyarrow.json
from pyarrow.tests.util import (FSProtocolClass, ProxyHandler,
def _do_list_all_dirs(basedir, path_so_far, result):
    for f in os.listdir(basedir):
        true_nested = os.path.join(basedir, f)
        if os.path.isdir(true_nested):
            norm_nested = posixpath.join(path_so_far, f)
            if _has_subdirs(true_nested):
                _do_list_all_dirs(true_nested, norm_nested, result)
            else:
                result.append(norm_nested)