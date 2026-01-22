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
def _GetTestExecutionStatus(self, test_id):
    """Fetch the TestExecution state of a specific test within a matrix.

    This method is only intended to be used for a TestMatrix with exactly one
    supported TestExecution. It would be inefficient to use it iteratively on
    a larger TestMatrix.

    Args:
      test_id: ID of the TestExecution status to find.

    Returns:
      The TestExecution message matching the unique test_id.
    """
    matrix = self.GetTestMatrixStatus()
    for test in matrix.testExecutions:
        if test.id == test_id:
            return test
    raise exceptions.TestExecutionNotFoundError(test_id, self.matrix_id)