from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
from apitools.base.py import exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.dataflow import apis
from googlecloudsdk.api_lib.dataflow import exceptions as dataflow_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetErrorMessage(error):
    """Extract the error message from an HTTPError.

  Args:
    error: The error exceptions.HttpError thrown by the API client.

  Returns:
    A string describing the error.
  """
    try:
        content_obj = json.loads(error.content)
        return content_obj.get('error', {}).get('message', '')
    except ValueError:
        log.err.Print(error.response)
        return 'Unknown error'