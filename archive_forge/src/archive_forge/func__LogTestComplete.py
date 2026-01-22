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
def _LogTestComplete(self, matrix_state):
    """Let the user know that their test matrix has completed running."""
    log.info('Test matrix completed in state: {0}'.format(matrix_state))
    log.status.Print('\n{0} testing complete.'.format(self._test_type.capitalize()))