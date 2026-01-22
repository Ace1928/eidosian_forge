import os
import shutil
import subprocess
import sys
import fixtures
import testresources
import testtools
from testtools import content
from pbr import options
def _discard_testpackage(self):
    for k in list(sys.modules):
        if k == 'pbr_testpackage' or k.startswith('pbr_testpackage.'):
            del sys.modules[k]