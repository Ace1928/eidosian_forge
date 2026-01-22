from __future__ import absolute_import, print_function, division
import sys
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import pytest
from petl.test.helpers import ieq
from petl.util import nrows, look
from petl.io.xml import fromxml, toxml
from petl.compat import urlopen
def _write_test_file(data, pre='', pos=''):
    content = pre + '<table>' + data + pos + '</table>'
    return _write_temp_file(content)