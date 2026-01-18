import bz2
import functools
import gzip
import itertools
import os
import tempfile
import threading
import time
import warnings
from io import BytesIO
from os.path import exists
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from packaging.version import Version
from nibabel.testing import (
from ..casting import OK_FLOATS, floor_log2, sctypes, shared_range, type_info
from ..openers import BZ2File, ImageOpener, Opener
from ..optpkg import optional_package
from ..tmpdirs import InTemporaryDirectory
from ..volumeutils import (
def test_write_zeros():
    bio = BytesIO()
    write_zeros(bio, 10000)
    assert bio.getvalue() == b'\x00' * 10000
    bio.seek(0)
    bio.truncate(0)
    write_zeros(bio, 10000, 256)
    assert bio.getvalue() == b'\x00' * 10000
    bio.seek(0)
    bio.truncate(0)
    write_zeros(bio, 200, 256)
    assert bio.getvalue() == b'\x00' * 200