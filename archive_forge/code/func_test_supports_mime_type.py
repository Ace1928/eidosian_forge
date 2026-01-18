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
@pytest.mark.xfail(cairo_version() < 11200, reason='Cairo version too low')
def test_supports_mime_type():
    assert PDFSurface(None, 1, 1).supports_mime_type('image/jpeg') is True
    surface = ImageSurface(cairocffi.FORMAT_A8, 1, 1)
    assert surface.supports_mime_type('image/jpeg') is False