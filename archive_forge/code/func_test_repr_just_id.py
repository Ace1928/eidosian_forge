from doctest import ELLIPSIS
from pprint import pformat
import sys
import _thread
import unittest
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.matchers import (
from testtools.testcase import (
from testtools.testresult.doubles import (
from testtools.tests.helpers import (
from testtools.tests.samplecases import (
def test_repr_just_id(self):
    test = PlaceHolder('test id')
    self.assertEqual("<testtools.testcase.PlaceHolder('addSuccess', %s, {})>" % repr(test.id()), repr(test))