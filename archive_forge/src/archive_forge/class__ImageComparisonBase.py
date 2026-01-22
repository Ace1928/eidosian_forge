import contextlib
import functools
import inspect
import os
from platform import uname
from pathlib import Path
import shutil
import string
import sys
import warnings
from packaging.version import parse as parse_version
import matplotlib.style
import matplotlib.units
import matplotlib.testing
from matplotlib import _pylab_helpers, cbook, ft2font, pyplot as plt, ticker
from .compare import comparable_formats, compare_images, make_test_filename
from .exceptions import ImageComparisonFailure
class _ImageComparisonBase:
    """
    Image comparison base class

    This class provides *just* the comparison-related functionality and avoids
    any code that would be specific to any testing framework.
    """

    def __init__(self, func, tol, remove_text, savefig_kwargs):
        self.func = func
        self.baseline_dir, self.result_dir = _image_directories(func)
        self.tol = tol
        self.remove_text = remove_text
        self.savefig_kwargs = savefig_kwargs

    def copy_baseline(self, baseline, extension):
        baseline_path = self.baseline_dir / baseline
        orig_expected_path = baseline_path.with_suffix(f'.{extension}')
        if extension == 'eps' and (not orig_expected_path.exists()):
            orig_expected_path = orig_expected_path.with_suffix('.pdf')
        expected_fname = make_test_filename(self.result_dir / orig_expected_path.name, 'expected')
        try:
            with contextlib.suppress(OSError):
                os.remove(expected_fname)
            try:
                if 'microsoft' in uname().release.lower():
                    raise OSError
                os.symlink(orig_expected_path, expected_fname)
            except OSError:
                shutil.copyfile(orig_expected_path, expected_fname)
        except OSError as err:
            raise ImageComparisonFailure(f'Missing baseline image {expected_fname} because the following file cannot be accessed: {orig_expected_path}') from err
        return expected_fname

    def compare(self, fig, baseline, extension, *, _lock=False):
        __tracebackhide__ = True
        if self.remove_text:
            remove_ticks_and_titles(fig)
        actual_path = (self.result_dir / baseline).with_suffix(f'.{extension}')
        kwargs = self.savefig_kwargs.copy()
        if extension == 'pdf':
            kwargs.setdefault('metadata', {'Creator': None, 'Producer': None, 'CreationDate': None})
        lock = cbook._lock_path(actual_path) if _lock else contextlib.nullcontext()
        with lock:
            try:
                fig.savefig(actual_path, **kwargs)
            finally:
                plt.close(fig)
            expected_path = self.copy_baseline(baseline, extension)
            _raise_on_image_difference(expected_path, actual_path, self.tol)