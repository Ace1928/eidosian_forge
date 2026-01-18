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
def test_supplies_timestamps(self):
    test = PlaceHolder('foo', details={}, timestamps=['A', 'B'])
    result = ExtendedTestResult()
    test.run(result)
    self.assertEqual([('time', 'A'), ('tags', set(), set()), ('startTest', test), ('time', 'B'), ('addSuccess', test), ('stopTest', test), ('tags', set(), set())], result._events)