import os
import tarfile
import tempfile
import warnings
from io import BytesIO
from shutil import copy2, copytree, rmtree
from .. import osutils
from .. import revision as _mod_revision
from .. import transform
from ..controldir import ControlDir
from ..export import export
from ..upstream_import import (NotArchiveType, ZipFileWrapper,
from . import TestCaseInTempDir, TestCaseWithTransport
from .features import UnicodeFilenameFeature
def test_top_path(self):
    self.assertEqual(top_path('ab/b/c'), 'ab')
    self.assertEqual(top_path('etc'), 'etc')
    self.assertEqual(top_path('project-0.1'), 'project-0.1')