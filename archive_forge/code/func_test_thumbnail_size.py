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
@pytest.mark.xfail(cairo_version() < 11504, reason='Cairo version too low')
def test_thumbnail_size():
    file_obj = io.BytesIO()
    surface = PDFSurface(file_obj, 1, 1)
    surface.set_thumbnail_size(1, 1)
    surface.finish()
    pdf_bytes1 = file_obj.getvalue()
    file_obj = io.BytesIO()
    surface = PDFSurface(file_obj, 1, 1)
    surface.set_thumbnail_size(9, 9)
    surface.finish()
    pdf_bytes2 = file_obj.getvalue()
    assert len(pdf_bytes1) < len(pdf_bytes2)