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
def test_vanilla(self):
    with InTemporaryDirectory():
        with ImageOpener('test.gz', 'w') as fobj:
            assert hasattr(fobj.fobj, 'compress')
        with ImageOpener('test.mgz', 'w') as fobj:
            assert hasattr(fobj.fobj, 'compress')