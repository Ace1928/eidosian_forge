import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetBundleContentsFolderPath(self):
    """Returns the qualified path to the bundle's contents folder. E.g.
    Chromium.app/Contents or Foo.bundle/Versions/A. Only valid for bundles."""
    if self.isIOS:
        return self.GetWrapperName()
    assert self._IsBundle()
    if self.spec['type'] == 'shared_library':
        return os.path.join(self.GetWrapperName(), 'Versions', self.GetFrameworkVersion())
    else:
        return os.path.join(self.GetWrapperName(), 'Contents')