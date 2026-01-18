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
def test_unbounded_recording_surface():
    text_pixels, recorded_pixels = _recording_surface_common(None)
    assert recorded_pixels == text_pixels