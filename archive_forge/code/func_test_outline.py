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
def test_outline():
    file_obj = io.BytesIO()
    surface = PDFSurface(file_obj, 1, 1)
    outline = surface.add_outline(PDF_OUTLINE_ROOT, 'title 1', 'page=1 pos=[1 1]', PDF_OUTLINE_FLAG_OPEN & PDF_OUTLINE_FLAG_BOLD)
    surface.add_outline(outline, 'title 2', 'page=1 pos=[1 1]')
    surface.finish()
    pdf = pikepdf.Pdf.open(file_obj)
    outline = pdf.open_outline()
    assert outline.root[0].title == 'title 1'
    assert outline.root[0].children[0].title == 'title 2'