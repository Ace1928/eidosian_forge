import datetime
import re
import sys
import threading
import time
import traceback
import unittest
from xml.sax import saxutils
from absl.testing import _pretty_print_reporter
class _TestCaseResult(object):
    """Private helper for _TextAndXMLTestResult that represents a test result.

  Attributes:
    test: A TestCase instance of an individual test method.
    name: The name of the individual test method.
    full_class_name: The full name of the test class.
    run_time: The duration (in seconds) it took to run the test.
    start_time: Epoch relative timestamp of when test started (in seconds)
    errors: A list of error 4-tuples. Error tuple entries are
        1) a string identifier of either "failure" or "error"
        2) an exception_type
        3) an exception_message
        4) a string version of a sys.exc_info()-style tuple of values
           ('error', err[0], err[1], self._exc_info_to_string(err))
           If the length of errors is 0, then the test is either passed or
           skipped.
    skip_reason: A string explaining why the test was skipped.
  """

    def __init__(self, test):
        self.run_time = -1
        self.start_time = -1
        self.skip_reason = None
        self.errors = []
        self.test = test
        test_desc = test.id() or str(test)
        match = _CLASS_OR_MODULE_LEVEL_TEST_DESC_REGEX.match(test_desc)
        if match:
            name = match.group(1)
            full_class_name = match.group(2)
        else:
            class_name = unittest.util.strclass(test.__class__)
            if isinstance(test, unittest.case._SubTest):
                class_name = unittest.util.strclass(test.test_case.__class__)
            if test_desc.startswith(class_name + '.'):
                name = test_desc[len(class_name) + 1:]
                full_class_name = class_name
            else:
                parts = test_desc.rsplit('.', 1)
                name = parts[-1]
                full_class_name = parts[0] if len(parts) == 2 else ''
        self.name = _escape_xml_attr(name)
        self.full_class_name = _escape_xml_attr(full_class_name)

    def set_run_time(self, time_in_secs):
        self.run_time = time_in_secs

    def set_start_time(self, time_in_secs):
        self.start_time = time_in_secs

    def print_xml_summary(self, stream):
        """Prints an XML Summary of a TestCase.

    Status and result are populated as per JUnit XML test result reporter.
    A test that has been skipped will always have a skip reason,
    as every skip method in Python's unittest requires the reason arg to be
    passed.

    Args:
      stream: output stream to write test report XML to
    """
        if self.skip_reason is None:
            status = 'run'
            result = 'completed'
        else:
            status = 'notrun'
            result = 'suppressed'
        test_case_attributes = [('name', '%s' % self.name), ('status', '%s' % status), ('result', '%s' % result), ('time', '%.3f' % self.run_time), ('classname', self.full_class_name), ('timestamp', _iso8601_timestamp(self.start_time))]
        _print_xml_element_header('testcase', test_case_attributes, stream, '  ')
        self._print_testcase_details(stream)
        stream.write('  </testcase>\n')

    def _print_testcase_details(self, stream):
        for error in self.errors:
            outcome, exception_type, message, error_msg = error
            message = _escape_xml_attr(_safe_str(message))
            exception_type = _escape_xml_attr(str(exception_type))
            error_msg = _escape_cdata(error_msg)
            stream.write('  <%s message="%s" type="%s"><![CDATA[%s]]></%s>\n' % (outcome, message, exception_type, error_msg, outcome))