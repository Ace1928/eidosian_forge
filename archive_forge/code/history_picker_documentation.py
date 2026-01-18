from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
Gets the history id associated with a given history name.

    All the test executions for the same app should be in the same history. This
    method will try to find an existing history for the application or create
    one if this is the first time a particular history_name has been seen.

    Args:
       history_name: string containing the history's name (if the user supplied
         one), else None.

    Returns:
      The id of the history to publish results to.
    