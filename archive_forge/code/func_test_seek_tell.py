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
def test_seek_tell():
    bio = BytesIO()
    in_files = [bio, 'test.bin', 'test.gz', 'test.bz2']
    if HAVE_ZSTD:
        in_files += ['test.zst']
    start = 10
    end = 100
    diff = end - start
    tail = 7
    with InTemporaryDirectory():
        for in_file, write0 in itertools.product(in_files, (False, True)):
            st = functools.partial(seek_tell, write0=write0)
            bio.seek(0)
            with ImageOpener(in_file, 'wb') as fobj:
                assert fobj.tell() == 0
                st(fobj, 0)
                assert fobj.tell() == 0
                fobj.write(b'\x01' * start)
                assert fobj.tell() == start
                if not write0 and in_file in ('test.bz2', 'test.zst'):
                    fobj.write(b'\x00' * diff)
                else:
                    st(fobj, end)
                    assert fobj.tell() == end
                fobj.write(b'\x02' * tail)
            bio.seek(0)
            with ImageOpener(in_file, 'rb') as fobj:
                assert fobj.tell() == 0
                st(fobj, 0)
                assert fobj.tell() == 0
                st(fobj, start)
                assert fobj.tell() == start
                st(fobj, end)
                assert fobj.tell() == end
                st(fobj, 0)
            bio.seek(0)
            with ImageOpener(in_file, 'rb') as fobj:
                assert fobj.read() == b'\x01' * start + b'\x00' * diff + b'\x02' * tail
        input_files = ['test2.gz', 'test2.bz2']
        if HAVE_ZSTD:
            input_files += ['test2.zst']
        for in_file in input_files:
            with ImageOpener(in_file, 'wb') as fobj:
                fobj.write(b'g' * 10)
                assert fobj.tell() == 10
                seek_tell(fobj, 10)
                assert fobj.tell() == 10
                with pytest.raises(OSError):
                    seek_tell(fobj, 5)
            with ImageOpener(in_file, 'rb') as fobj:
                seek_tell(fobj, 10)
                seek_tell(fobj, 0)
            with ImageOpener(in_file, 'rb') as fobj:
                assert fobj.read() == b'g' * 10