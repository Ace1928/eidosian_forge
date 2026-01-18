import contextlib
import hashlib
import os
import time
import unittest
from gzip import GzipFile
from io import BytesIO, UnsupportedOperation
from unittest import mock
import pytest
from packaging.version import Version
from ..deprecator import ExpiredDeprecationError
from ..openers import HAVE_INDEXED_GZIP, BZ2File, DeterministicGzipFile, ImageOpener, Opener
from ..optpkg import optional_package
from ..tmpdirs import InTemporaryDirectory
def test_set_extensions():
    with InTemporaryDirectory():
        with Opener('test.gz', 'w') as fobj:
            assert hasattr(fobj.fobj, 'compress')
        with Opener('test.glrph', 'w') as fobj:
            assert not hasattr(fobj.fobj, 'compress')

        class MyOpener(Opener):
            compress_ext_map = Opener.compress_ext_map.copy()
            compress_ext_map['.glrph'] = Opener.gz_def
        with MyOpener('test.glrph', 'w') as fobj:
            assert hasattr(fobj.fobj, 'compress')