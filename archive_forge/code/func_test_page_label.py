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
def test_page_label():
    file_obj = io.BytesIO()
    surface = PDFSurface(file_obj, 1, 1)
    surface.set_page_label('abc')
    surface.finish()
    pdf = pikepdf.Pdf.open(file_obj)
    assert pdf.pages[0].label == 'abc'