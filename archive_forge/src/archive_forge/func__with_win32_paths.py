import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def _with_win32_paths(self):
    self.overrideAttr(urlutils, 'local_path_from_url', urlutils._win32_local_path_from_url)
    self.overrideAttr(urlutils, 'MIN_ABS_FILEURL_LENGTH', urlutils.WIN32_MIN_ABS_FILEURL_LENGTH)
    self.overrideAttr(osutils, 'abspath', osutils._win32_abspath)
    self.overrideAttr(osutils, 'normpath', osutils._win32_normpath)
    self.overrideAttr(osutils, 'pathjoin', osutils._win32_pathjoin)
    self.overrideAttr(osutils, 'split', osutils.ntpath.split)
    self.overrideAttr(osutils, 'MIN_ABS_PATHLENGTH', 3)