from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import fnmatch
import json
from googlecloudsdk.command_lib.anthos.config.sync.common import exceptions
from googlecloudsdk.command_lib.anthos.config.sync.common import utils
from googlecloudsdk.core import log
def _GetConditionForType(obj, condition_type):
    """Return the object condition for the given type.

  Args:
    obj: The json object that represents a RepoSync|RootSync CR.
    condition_type: Condition type.

  Returns:
    The condition for the given type.
  """
    conditions = _GetPathValue(obj, ['status', 'conditions'])
    if not conditions:
        return False
    for condition in conditions:
        if condition['type'] == condition_type:
            return condition
    return None