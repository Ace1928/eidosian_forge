import copy
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock
from cycler import cycler, Cycler
import pytest
import matplotlib as mpl
from matplotlib import _api, _c_internal_utils
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.rcsetup import (
def test_rcparams(tmpdir):
    mpl.rc('text', usetex=False)
    mpl.rc('lines', linewidth=22)
    usetex = mpl.rcParams['text.usetex']
    linewidth = mpl.rcParams['lines.linewidth']
    rcpath = Path(tmpdir) / 'test_rcparams.rc'
    rcpath.write_text('lines.linewidth: 33', encoding='utf-8')
    with mpl.rc_context(rc={'text.usetex': not usetex}):
        assert mpl.rcParams['text.usetex'] == (not usetex)
    assert mpl.rcParams['text.usetex'] == usetex
    with mpl.rc_context(fname=rcpath):
        assert mpl.rcParams['lines.linewidth'] == 33
    assert mpl.rcParams['lines.linewidth'] == linewidth
    with mpl.rc_context(fname=rcpath, rc={'lines.linewidth': 44}):
        assert mpl.rcParams['lines.linewidth'] == 44
    assert mpl.rcParams['lines.linewidth'] == linewidth

    @mpl.rc_context({'lines.linewidth': 44})
    def func():
        assert mpl.rcParams['lines.linewidth'] == 44
    func()
    func()
    mpl.rc_file(rcpath)
    assert mpl.rcParams['lines.linewidth'] == 33