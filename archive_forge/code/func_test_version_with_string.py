import os
import subprocess
import sys
from distutils import version
import breezy
from .. import tests
def test_version_with_string(self):
    lv = version.LooseVersion
    self.assertTrue(lv('0.9.4.1') < lv('0.17.beta1'))
    self.assertTrue(lv('0.9.6.3') < lv('0.10'))