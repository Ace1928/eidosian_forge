from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.py import encoding
from googlecloudsdk.command_lib.logs import stream
import six
def _GetTaskName(log_entry):
    """Reads the label attributes of the given log entry."""
    resource_labels = {} if not log_entry.resource else _ToDict(log_entry.resource.labels)
    return 'unknown_task' if not resource_labels.get('task_name') else resource_labels['task_name']