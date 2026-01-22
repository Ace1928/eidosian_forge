from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DryRunChanges(_messages.Message):
    """Message describing proposed changes from dry run.

  Fields:
    hasChanges: Whether there are changes.
    textOutput: Formatted output of the changes. Same format as the terraform
      plan output.
  """
    hasChanges = _messages.BooleanField(1)
    textOutput = _messages.StringField(2)