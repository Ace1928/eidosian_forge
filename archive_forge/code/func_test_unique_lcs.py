import os
import shutil
import tempfile
import unittest
import patiencediff
from . import _patiencediff_py
def test_unique_lcs(self):
    try:
        from ._patiencediff_c import unique_lcs_c
    except ImportError:
        from ._patiencediff_py import unique_lcs_py
        self.assertIs(unique_lcs_py, patiencediff.unique_lcs)
    else:
        self.assertIs(unique_lcs_c, patiencediff.unique_lcs)