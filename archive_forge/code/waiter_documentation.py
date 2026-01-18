from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import time
from apitools.base.py import encoding
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
import six
Overrides to get the response from the completed operation.

    Args:
      operation: api_name_messages.Operation.

    Returns:
      the 'response' field of the Operation.
    