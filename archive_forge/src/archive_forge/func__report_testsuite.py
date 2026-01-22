import random
import unittest
import xmlrunner
from __future__ import absolute_import
import os
import sys
import time
from unittest import TestResult, TextTestResult, TextTestRunner
import xml.dom.minidom
def _report_testsuite(suite_name, tests, xml_document):
    """Appends the testsuite section to the XML document."""
    testsuite = xml_document.createElement('testsuite')
    xml_document.appendChild(testsuite)
    testsuite.setAttribute('name', str(suite_name))
    testsuite.setAttribute('tests', str(len(tests)))
    testsuite.setAttribute('time', '%.3f' % sum([e.get_elapsed_time() for e in tests]))
    failures = len([1 for e in tests if e.outcome == _TestInfo.FAILURE])
    testsuite.setAttribute('failures', str(failures))
    errors = len([1 for e in tests if e.outcome == _TestInfo.ERROR])
    testsuite.setAttribute('errors', str(errors))
    return testsuite