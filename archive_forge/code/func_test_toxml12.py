from __future__ import absolute_import, print_function, division
import sys
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import pytest
from petl.test.helpers import ieq
from petl.util import nrows, look
from petl.io.xml import fromxml, toxml
from petl.compat import urlopen
def test_toxml12():
    _check_toxml(_TABLE1, _TABLE1, check=('.//tr/td', _ATTRIB_COLS), style='attribute', root='table', head='thead/tr/th', rows='tbody/tr/td')