import datetime
import re
import sys
import threading
import time
import traceback
import unittest
from xml.sax import saxutils
from absl.testing import _pretty_print_reporter
class _TestSuiteResult(object):
    """Private helper for _TextAndXMLTestResult."""

    def __init__(self):
        self.suites = {}
        self.failure_counts = {}
        self.error_counts = {}
        self.overall_start_time = -1
        self.overall_end_time = -1
        self._testsuites_properties = {}

    def add_test_case_result(self, test_case_result):
        suite_name = type(test_case_result.test).__name__
        if suite_name == '_ErrorHolder':
            suite_name = test_case_result.full_class_name.rsplit('.')[-1]
        if isinstance(test_case_result.test, unittest.case._SubTest):
            suite_name = type(test_case_result.test.test_case).__name__
        self._setup_test_suite(suite_name)
        self.suites[suite_name].append(test_case_result)
        for error in test_case_result.errors:
            if error[0] == 'failure':
                self.failure_counts[suite_name] += 1
                break
            elif error[0] == 'error':
                self.error_counts[suite_name] += 1
                break

    def print_xml_summary(self, stream):
        overall_test_count = sum((len(x) for x in self.suites.values()))
        overall_failures = sum(self.failure_counts.values())
        overall_errors = sum(self.error_counts.values())
        overall_attributes = [('name', ''), ('tests', '%d' % overall_test_count), ('failures', '%d' % overall_failures), ('errors', '%d' % overall_errors), ('time', '%.3f' % (self.overall_end_time - self.overall_start_time)), ('timestamp', _iso8601_timestamp(self.overall_start_time))]
        _print_xml_element_header('testsuites', overall_attributes, stream)
        if self._testsuites_properties:
            stream.write('    <properties>\n')
            for name, value in sorted(self._testsuites_properties.items()):
                stream.write('      <property name="%s" value="%s"></property>\n' % (_escape_xml_attr(name), _escape_xml_attr(str(value))))
            stream.write('    </properties>\n')
        for suite_name in self.suites:
            suite = self.suites[suite_name]
            suite_end_time = max((x.start_time + x.run_time for x in suite))
            suite_start_time = min((x.start_time for x in suite))
            failures = self.failure_counts[suite_name]
            errors = self.error_counts[suite_name]
            suite_attributes = [('name', '%s' % suite_name), ('tests', '%d' % len(suite)), ('failures', '%d' % failures), ('errors', '%d' % errors), ('time', '%.3f' % (suite_end_time - suite_start_time)), ('timestamp', _iso8601_timestamp(suite_start_time))]
            _print_xml_element_header('testsuite', suite_attributes, stream)
            for test_case_result in sorted(suite, key=lambda t: t.name):
                test_case_result.print_xml_summary(stream)
            stream.write('</testsuite>\n')
        stream.write('</testsuites>\n')

    def _setup_test_suite(self, suite_name):
        """Adds a test suite to the set of suites tracked by this test run.

    Args:
      suite_name: string, The name of the test suite being initialized.
    """
        if suite_name in self.suites:
            return
        self.suites[suite_name] = []
        self.failure_counts[suite_name] = 0
        self.error_counts[suite_name] = 0

    def set_end_time(self, timestamp_in_secs):
        """Sets the start timestamp of this test suite.

    Args:
      timestamp_in_secs: timestamp in seconds since epoch
    """
        self.overall_end_time = timestamp_in_secs

    def set_start_time(self, timestamp_in_secs):
        """Sets the end timestamp of this test suite.

    Args:
      timestamp_in_secs: timestamp in seconds since epoch
    """
        self.overall_start_time = timestamp_in_secs