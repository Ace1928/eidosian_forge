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
def test_svg_surface():
    assert set(SVGSurface.get_versions()) >= set([cairocffi.SVG_VERSION_1_1, cairocffi.SVG_VERSION_1_2])
    assert SVGSurface.version_to_string(cairocffi.SVG_VERSION_1_1) == 'SVG 1.1'
    with pytest.raises(TypeError):
        SVGSurface.version_to_string('SVG_VERSION_42')
    with pytest.raises(ValueError):
        SVGSurface.version_to_string(42)
    with temp_directory() as tempdir:
        filename = os.path.join(tempdir, 'foo.svg')
        filename_bytes = filename.encode(sys.getfilesystemencoding())
        file_obj = io.BytesIO()
        for target in [filename, filename_bytes, file_obj, None]:
            SVGSurface(target, 123, 432).finish()
        with open(filename, 'rb') as fd:
            assert fd.read().startswith(b'<?xml')
        with open(filename_bytes, 'rb') as fd:
            assert fd.read().startswith(b'<?xml')
        svg_bytes = file_obj.getvalue()
        assert svg_bytes.startswith(b'<?xml')
        assert b'viewBox="0 0 123 432"' in svg_bytes
    surface = SVGSurface(None, 1, 1)
    surface.restrict_to_version(cairocffi.SVG_VERSION_1_1)