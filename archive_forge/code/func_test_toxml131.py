from __future__ import absolute_import, print_function, division
import sys
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import pytest
from petl.test.helpers import ieq
from petl.util import nrows, look
from petl.io.xml import fromxml, toxml
from petl.compat import urlopen
def test_toxml131():
    _check_toxml(_TABLE1, _TABLE1, check=('.//tr', ('th', 'td')), style=' <tr><td>{ABCD}</td><td>{N123}</td></tr>\n', root='table', head='thead/tr/td', rows='tbody')