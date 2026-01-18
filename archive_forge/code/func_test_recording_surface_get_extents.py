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
def test_recording_surface_get_extents():
    for extents in [None, (0, 0, 140, 80)]:
        surface = RecordingSurface(cairocffi.CONTENT_COLOR_ALPHA, extents)
        assert surface.get_extents() == extents