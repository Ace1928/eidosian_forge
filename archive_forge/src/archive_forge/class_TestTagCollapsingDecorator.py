import csv
import datetime
import sys
import unittest
from io import StringIO
import testtools
from testtools import TestCase
from testtools.content import TracebackContent, text_content
from testtools.testresult.doubles import ExtendedTestResult
import subunit
import iso8601
import subunit.test_results
class TestTagCollapsingDecorator(TestCase):

    def test_tags_collapsed_outside_of_tests(self):
        result = ExtendedTestResult()
        tag_collapser = subunit.test_results.TagCollapsingDecorator(result)
        tag_collapser.tags({'a'}, set())
        tag_collapser.tags({'b'}, set())
        tag_collapser.startTest(self)
        self.assertEqual([('tags', {'a', 'b'}, set()), ('startTest', self)], result._events)

    def test_tags_collapsed_outside_of_tests_are_flushed(self):
        result = ExtendedTestResult()
        tag_collapser = subunit.test_results.TagCollapsingDecorator(result)
        tag_collapser.startTestRun()
        tag_collapser.tags({'a'}, set())
        tag_collapser.tags({'b'}, set())
        tag_collapser.startTest(self)
        tag_collapser.addSuccess(self)
        tag_collapser.stopTest(self)
        tag_collapser.stopTestRun()
        self.assertEqual([('startTestRun',), ('tags', {'a', 'b'}, set()), ('startTest', self), ('addSuccess', self), ('stopTest', self), ('stopTestRun',)], result._events)

    def test_tags_forwarded_after_tests(self):
        test = subunit.RemotedTestCase('foo')
        result = ExtendedTestResult()
        tag_collapser = subunit.test_results.TagCollapsingDecorator(result)
        tag_collapser.startTestRun()
        tag_collapser.startTest(test)
        tag_collapser.addSuccess(test)
        tag_collapser.stopTest(test)
        tag_collapser.tags({'a'}, {'b'})
        tag_collapser.stopTestRun()
        self.assertEqual([('startTestRun',), ('startTest', test), ('addSuccess', test), ('stopTest', test), ('tags', {'a'}, {'b'}), ('stopTestRun',)], result._events)

    def test_tags_collapsed_inside_of_tests(self):
        result = ExtendedTestResult()
        tag_collapser = subunit.test_results.TagCollapsingDecorator(result)
        test = subunit.RemotedTestCase('foo')
        tag_collapser.startTest(test)
        tag_collapser.tags({'a'}, set())
        tag_collapser.tags({'b'}, {'a'})
        tag_collapser.tags({'c'}, set())
        tag_collapser.stopTest(test)
        self.assertEqual([('startTest', test), ('tags', {'b', 'c'}, {'a'}), ('stopTest', test)], result._events)

    def test_tags_collapsed_inside_of_tests_different_ordering(self):
        result = ExtendedTestResult()
        tag_collapser = subunit.test_results.TagCollapsingDecorator(result)
        test = subunit.RemotedTestCase('foo')
        tag_collapser.startTest(test)
        tag_collapser.tags(set(), {'a'})
        tag_collapser.tags({'a', 'b'}, set())
        tag_collapser.tags({'c'}, set())
        tag_collapser.stopTest(test)
        self.assertEqual([('startTest', test), ('tags', {'a', 'b', 'c'}, set()), ('stopTest', test)], result._events)

    def test_tags_sent_before_result(self):
        result = ExtendedTestResult()
        tag_collapser = subunit.test_results.TagCollapsingDecorator(result)
        test = subunit.RemotedTestCase('foo')
        tag_collapser.startTest(test)
        tag_collapser.tags({'a'}, set())
        tag_collapser.addSuccess(test)
        tag_collapser.stopTest(test)
        self.assertEqual([('startTest', test), ('tags', {'a'}, set()), ('addSuccess', test), ('stopTest', test)], result._events)