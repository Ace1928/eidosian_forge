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
def test_surface():
    surface = ImageSurface(cairocffi.FORMAT_ARGB32, 20, 30)
    similar = surface.create_similar(cairocffi.CONTENT_ALPHA, 4, 100)
    assert isinstance(similar, ImageSurface)
    assert similar.get_content() == cairocffi.CONTENT_ALPHA
    assert similar.get_format() == cairocffi.FORMAT_A8
    assert similar.get_width() == 4
    assert similar.get_height() == 100
    assert similar.has_show_text_glyphs() is False
    assert PDFSurface(None, 1, 1).has_show_text_glyphs() is True
    surface.copy_page()
    surface.show_page()
    surface.mark_dirty()
    surface.mark_dirty_rectangle(1, 2, 300, 12000)
    surface.flush()
    surface.set_device_offset(14, 3)
    assert surface.get_device_offset() == (14, 3)
    surface.set_fallback_resolution(15, 6)
    assert surface.get_fallback_resolution() == (15, 6)
    context = Context(surface)
    assert isinstance(context.get_target(), ImageSurface)
    surface_map = cairocffi.surfaces.SURFACE_TYPE_TO_CLASS
    try:
        del surface_map[cairocffi.SURFACE_TYPE_IMAGE]
        target = context.get_target()
        assert target._pointer == surface._pointer
        assert isinstance(target, Surface)
        assert not isinstance(target, ImageSurface)
    finally:
        surface_map[cairocffi.SURFACE_TYPE_IMAGE] = ImageSurface
    surface.finish()
    assert_raise_finished(surface.copy_page)
    assert_raise_finished(surface.show_page)
    assert_raise_finished(surface.set_device_offset, 1, 2)
    assert_raise_finished(surface.set_fallback_resolution, 3, 4)