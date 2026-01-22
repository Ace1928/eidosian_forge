from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import time
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def HandleUnsupportedExecutions(self, matrix):
    """Report unsupported device dimensions and return supported test list.

    Args:
      matrix: a TestMatrix message.

    Returns:
      A list of TestExecution messages which have supported dimensions.
    """
    states = self._messages.TestExecution.StateValueValuesEnum
    supported_tests = []
    unsupported_dimensions = set()
    for test in matrix.testExecutions:
        if test.state == states.UNSUPPORTED_ENVIRONMENT:
            unsupported_dimensions.add(_FormatInvalidDimension(test.environment))
        else:
            supported_tests.append(test)
    if unsupported_dimensions:
        log.status.Print('Some device dimensions are not compatible and will be skipped:\n  {d}'.format(d='\n  '.join(unsupported_dimensions)))
    log.status.Print('Firebase Test Lab will execute your {t} test on {n} device(s). More devices may be added later if flaky test attempts are specified.'.format(t=self._test_type, n=len(supported_tests)))
    return supported_tests