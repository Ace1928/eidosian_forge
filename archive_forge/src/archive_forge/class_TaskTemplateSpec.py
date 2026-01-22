from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TaskTemplateSpec(_messages.Message):
    """TaskTemplateSpec describes the data a task should have when created from
  a template.

  Fields:
    spec: Optional. Specification of the desired behavior of the task.
  """
    spec = _messages.MessageField('TaskSpec', 1)