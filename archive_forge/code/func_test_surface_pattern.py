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
def test_surface_pattern():
    surface = ImageSurface(cairocffi.FORMAT_A1, 1, 1)
    pattern = SurfacePattern(surface)
    surface_again = pattern.get_surface()
    assert surface_again is not surface
    assert surface_again._pointer == surface._pointer
    assert pattern.get_extend() == cairocffi.EXTEND_NONE
    pattern.set_extend(cairocffi.EXTEND_REPEAT)
    assert pattern.get_extend() == cairocffi.EXTEND_REPEAT
    assert pattern.get_filter() == cairocffi.FILTER_GOOD
    pattern.set_filter(cairocffi.FILTER_BEST)
    assert pattern.get_filter() == cairocffi.FILTER_BEST
    assert pattern.get_matrix() == Matrix()
    matrix = Matrix.init_rotate(0.5)
    pattern.set_matrix(matrix)
    assert pattern.get_matrix() == matrix
    assert pattern.get_matrix() != Matrix()