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
def test_pdf_surface():
    with temp_directory() as tempdir:
        filename = os.path.join(tempdir, 'foo.pdf')
        filename_bytes = filename.encode(sys.getfilesystemencoding())
        file_obj = io.BytesIO()
        for target in [filename, filename_bytes, file_obj, None]:
            surface = PDFSurface(target, 123, 432)
            surface.finish()
        with open(filename, 'rb') as fd:
            assert fd.read().startswith(b'%PDF')
        with open(filename_bytes, 'rb') as fd:
            assert fd.read().startswith(b'%PDF')
        pdf = pikepdf.Pdf.open(file_obj)
        assert pdf.pages[0]['/MediaBox'] == [0, 0, 123, 432]
        assert len(pdf.pages) == 1
    file_obj = io.BytesIO()
    surface = PDFSurface(file_obj, 1, 1)
    context = Context(surface)
    surface.set_size(12, 100)
    context.show_page()
    surface.set_size(42, 700)
    context.copy_page()
    surface.finish()
    pdf = pikepdf.Pdf.open(file_obj)
    assert '"/MediaBox": [ 0 0 1 1 ]' not in str(pdf.objects)
    assert pdf.pages[0]['/MediaBox'] == [0, 0, 12, 100]
    assert pdf.pages[1]['/MediaBox'] == [0, 0, 42, 700]
    assert len(pdf.pages) == 2