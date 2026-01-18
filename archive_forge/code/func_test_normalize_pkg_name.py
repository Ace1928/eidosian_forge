import os
import shutil
import sys
import tempfile
import unittest
from io import StringIO
from pecan.tests import PecanTestCase
def test_normalize_pkg_name(self):
    from pecan.scaffolds import PecanScaffold
    s = PecanScaffold()
    assert s.normalize_pkg_name('sam') == 'sam'
    assert s.normalize_pkg_name('sam1') == 'sam1'
    assert s.normalize_pkg_name('sam_') == 'sam_'
    assert s.normalize_pkg_name('Sam') == 'sam'
    assert s.normalize_pkg_name('SAM') == 'sam'
    assert s.normalize_pkg_name('sam ') == 'sam'
    assert s.normalize_pkg_name(' sam') == 'sam'
    assert s.normalize_pkg_name('sam$') == 'sam'
    assert s.normalize_pkg_name('sam-sam') == 'samsam'