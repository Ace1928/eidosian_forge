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
@pytest.mark.xfail(cairo_version() < 11000, reason='Cairo version too low')
def test_pdf_versions():
    assert set(PDFSurface.get_versions()) >= set([cairocffi.PDF_VERSION_1_4, cairocffi.PDF_VERSION_1_5])
    assert PDFSurface.version_to_string(cairocffi.PDF_VERSION_1_4) == 'PDF 1.4'
    with pytest.raises(TypeError):
        PDFSurface.version_to_string('PDF_VERSION_42')
    with pytest.raises(ValueError):
        PDFSurface.version_to_string(42)
    file_obj = io.BytesIO()
    PDFSurface(file_obj, 1, 1).finish()
    assert file_obj.getvalue().startswith((b'%PDF-1.5', b'%PDF-1.7'))
    file_obj = io.BytesIO()
    surface = PDFSurface(file_obj, 1, 1)
    surface.restrict_to_version(cairocffi.PDF_VERSION_1_4)
    surface.finish()
    assert file_obj.getvalue().startswith(b'%PDF-1.4')