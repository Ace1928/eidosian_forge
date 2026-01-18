import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import weakref
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.testing.decorators import check_figures_equal
@pytest.mark.parametrize('writer, frame_format, output', gen_writers())
@pytest.mark.parametrize('anim', [dict(klass=dict)], indirect=['anim'])
def test_save_animation_smoketest(tmpdir, writer, frame_format, output, anim):
    if frame_format is not None:
        plt.rcParams['animation.frame_format'] = frame_format
    anim = animation.FuncAnimation(**anim)
    dpi = None
    codec = None
    if writer == 'ffmpeg':
        anim._fig.set_size_inches((10.85, 9.21))
        dpi = 100.0
        codec = 'h264'
    with tmpdir.as_cwd():
        anim.save(output, fps=30, writer=writer, bitrate=500, dpi=dpi, codec=codec)
    del anim