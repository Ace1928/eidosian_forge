from io import BytesIO, StringIO
import gc
import multiprocessing
import os
from pathlib import Path
from PIL import Image
import shutil
import subprocess
import sys
import warnings
import numpy as np
import pytest
from matplotlib.font_manager import (
from matplotlib import cbook, ft2font, pyplot as plt, rc_context, figure as mfigure
@pytest.mark.skipif(sys.platform != 'win32', reason='Windows only')
def test_user_fonts_win32():
    if not (os.environ.get('APPVEYOR') or os.environ.get('TF_BUILD')):
        pytest.xfail("This test should only run on CI (appveyor or azure) as the developer's font directory should remain unchanged.")
    pytest.xfail('We need to update the registry for this test to work')
    font_test_file = 'mpltest.ttf'
    fonts = findSystemFonts()
    if any((font_test_file in font for font in fonts)):
        pytest.skip(f'{font_test_file} already exists in system fonts')
    user_fonts_dir = MSUserFontDirectories[0]
    os.makedirs(user_fonts_dir)
    shutil.copy(Path(__file__).parent / font_test_file, user_fonts_dir)
    fonts = findSystemFonts()
    assert any((font_test_file in font for font in fonts))