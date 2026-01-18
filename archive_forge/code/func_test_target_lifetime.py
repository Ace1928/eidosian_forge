import array
import base64
import contextlib
import gc
import io
import math
import os
import shutil
import sys
import tempfile
import cairocffi
import pikepdf
import pytest
from . import (
def test_target_lifetime():
    if not hasattr(sys, 'getrefcount'):
        pytest.xfail()
    gc.collect()
    target = io.BytesIO()
    initial_refcount = sys.getrefcount(target)
    assert len(cairocffi.surfaces.KeepAlive.instances) == 0
    surface = PDFSurface(target, 100, 100)
    assert len(cairocffi.surfaces.KeepAlive.instances) == 1
    assert sys.getrefcount(target) == initial_refcount + 1
    del surface
    gc.collect()
    assert len(cairocffi.surfaces.KeepAlive.instances) == 0
    assert sys.getrefcount(target) == initial_refcount