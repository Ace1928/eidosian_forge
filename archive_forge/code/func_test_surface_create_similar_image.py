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
def test_surface_create_similar_image():
    surface = ImageSurface(cairocffi.FORMAT_ARGB32, 20, 30)
    similar = surface.create_similar_image(cairocffi.FORMAT_A8, 4, 100)
    assert isinstance(similar, ImageSurface)
    assert similar.get_content() == cairocffi.CONTENT_ALPHA
    assert similar.get_format() == cairocffi.FORMAT_A8
    assert similar.get_width() == 4
    assert similar.get_height() == 100