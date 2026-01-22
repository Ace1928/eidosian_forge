from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
import time
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from six.moves.urllib import parse
import uritemplate
class ToolResultsIds(collections.namedtuple('ToolResultsIds', ['history_id', 'execution_id'])):
    """A tuple to hold the history & execution IDs returned from Tool Results.

  Fields:
    history_id: a string with the Tool Results history ID to publish to.
    execution_id: a string with the ID of the Tool Results execution.
  """