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
def _test_threading():
    import threading
    from matplotlib.ft2font import LOAD_NO_HINTING
    import matplotlib.font_manager as fm
    N = 10
    b = threading.Barrier(N)

    def bad_idea(n):
        b.wait()
        for j in range(100):
            font = fm.get_font(fm.findfont('DejaVu Sans'))
            font.set_text(str(n), 0.0, flags=LOAD_NO_HINTING)
    threads = [threading.Thread(target=bad_idea, name=f'bad_thread_{j}', args=(j,)) for j in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()