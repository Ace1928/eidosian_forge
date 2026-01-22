import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def _with_posix_paths(self):
    self.overrideAttr(urlutils, 'local_path_from_url', urlutils._posix_local_path_from_url)
    self.overrideAttr(urlutils, 'MIN_ABS_FILEURL_LENGTH', len('file:///'))
    self.overrideAttr(osutils, 'normpath', osutils._posix_normpath)
    self.overrideAttr(osutils, 'abspath', osutils.posixpath.abspath)
    self.overrideAttr(osutils, 'normpath', osutils._posix_normpath)
    self.overrideAttr(osutils, 'pathjoin', osutils.posixpath.join)
    self.overrideAttr(osutils, 'split', osutils.posixpath.split)
    self.overrideAttr(osutils, 'MIN_ABS_PATHLENGTH', 1)