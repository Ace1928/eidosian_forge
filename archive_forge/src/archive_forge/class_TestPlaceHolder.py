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
class TestPlaceHolder(TestCase):
    run_test_with = FullStackRunTest

    def makePlaceHolder(self, test_id='foo', short_description=None):
        return PlaceHolder(test_id, short_description)

    def test_id_comes_from_constructor(self):
        test = PlaceHolder('test id')
        self.assertEqual('test id', test.id())

    def test_shortDescription_is_id(self):
        test = PlaceHolder('test id')
        self.assertEqual(test.id(), test.shortDescription())

    def test_shortDescription_specified(self):
        test = PlaceHolder('test id', 'description')
        self.assertEqual('description', test.shortDescription())

    def test_testcase_is_hashable(self):
        test = hash(self)
        self.assertEqual(unittest.TestCase.__hash__(self), test)

    def test_testcase_equals_edgecase(self):
        self.assertFalse(self == _thread.RLock())

    def test_repr_just_id(self):
        test = PlaceHolder('test id')
        self.assertEqual("<testtools.testcase.PlaceHolder('addSuccess', %s, {})>" % repr(test.id()), repr(test))

    def test_repr_with_description(self):
        test = PlaceHolder('test id', 'description')
        self.assertEqual("<testtools.testcase.PlaceHolder('addSuccess', {!r}, {{}}, {!r})>".format(test.id(), test.shortDescription()), repr(test))

    def test_repr_custom_outcome(self):
        test = PlaceHolder('test id', outcome='addSkip')
        self.assertEqual("<testtools.testcase.PlaceHolder('addSkip', %r, {})>" % test.id(), repr(test))

    def test_counts_as_one_test(self):
        test = self.makePlaceHolder()
        self.assertEqual(1, test.countTestCases())

    def test_str_is_id(self):
        test = self.makePlaceHolder()
        self.assertEqual(test.id(), str(test))

    def test_runs_as_success(self):
        test = self.makePlaceHolder()
        log = []
        test.run(LoggingResult(log))
        self.assertEqual([('tags', set(), set()), ('startTest', test), ('addSuccess', test), ('stopTest', test), ('tags', set(), set())], log)

    def test_supplies_details(self):
        details = {'quux': None}
        test = PlaceHolder('foo', details=details)
        result = ExtendedTestResult()
        test.run(result)
        self.assertEqual([('tags', set(), set()), ('startTest', test), ('addSuccess', test, details), ('stopTest', test), ('tags', set(), set())], result._events)

    def test_supplies_timestamps(self):
        test = PlaceHolder('foo', details={}, timestamps=['A', 'B'])
        result = ExtendedTestResult()
        test.run(result)
        self.assertEqual([('time', 'A'), ('tags', set(), set()), ('startTest', test), ('time', 'B'), ('addSuccess', test), ('stopTest', test), ('tags', set(), set())], result._events)

    def test_call_is_run(self):
        test = self.makePlaceHolder()
        run_log = []
        test.run(LoggingResult(run_log))
        call_log = []
        test(LoggingResult(call_log))
        self.assertEqual(run_log, call_log)

    def test_runs_without_result(self):
        self.makePlaceHolder().run()

    def test_debug(self):
        self.makePlaceHolder().debug()

    def test_supports_tags(self):
        result = ExtendedTestResult()
        tags = {'foo', 'bar'}
        case = PlaceHolder('foo', tags=tags)
        case.run(result)
        self.assertEqual([('tags', tags, set()), ('startTest', case), ('addSuccess', case), ('stopTest', case), ('tags', set(), tags)], result._events)