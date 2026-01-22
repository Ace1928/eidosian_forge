import glob
import os
import re
import docutils.core
from osprofiler.tests import test
def _check_no_cr(self, tpl, raw):
    matches = re.findall('\r', raw)
    self.assertEqual(len(matches), 0, 'Found %s literal carriage returns in file %s' % (len(matches), tpl))