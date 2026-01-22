import glob
import os
import re
import docutils.core
from osprofiler.tests import test
def _check_lines_wrapping(self, tpl, raw):
    for i, line in enumerate(raw.split('\n')):
        if 'http://' in line or 'https://' in line:
            continue
        self.assertTrue(len(line) < 80, msg='%s:%d: Line limited to a maximum of 79 characters.' % (tpl, i + 1))