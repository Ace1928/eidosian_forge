from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt, style
from matplotlib.style.core import USER_LIBRARY_PATHS, STYLE_EXTENSION
def test_style_from_module(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.chdir(tmp_path)
    pkg_path = tmp_path / 'mpl_test_style_pkg'
    pkg_path.mkdir()
    (pkg_path / 'test_style.mplstyle').write_text('lines.linewidth: 42', encoding='utf-8')
    pkg_path.with_suffix('.mplstyle').write_text('lines.linewidth: 84', encoding='utf-8')
    mpl.style.use('mpl_test_style_pkg.test_style')
    assert mpl.rcParams['lines.linewidth'] == 42
    mpl.style.use('mpl_test_style_pkg.mplstyle')
    assert mpl.rcParams['lines.linewidth'] == 84
    mpl.style.use('./mpl_test_style_pkg.mplstyle')
    assert mpl.rcParams['lines.linewidth'] == 84