from __future__ import absolute_import, print_function, division
import sys
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import pytest
from petl.test.helpers import ieq
from petl.util import nrows, look
from petl.io.xml import fromxml, toxml
from petl.compat import urlopen
def _write_temp_file(content, out=None):
    with NamedTemporaryFile(delete=False, mode='wt') as f:
        f.write(content)
        res = f.name
        f.close()
    if out is not None:
        outf = sys.stderr if out else sys.stdout
        print('TEST %s:\n%s' % (res, content), file=outf)
    return res