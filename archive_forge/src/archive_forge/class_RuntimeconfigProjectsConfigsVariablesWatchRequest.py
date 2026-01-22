from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuntimeconfigProjectsConfigsVariablesWatchRequest(_messages.Message):
    """A RuntimeconfigProjectsConfigsVariablesWatchRequest object.

  Fields:
    name: The name of the variable to watch, in the format:
      `projects/[PROJECT_ID]/configs/[CONFIG_NAME]`
    watchVariableRequest: A WatchVariableRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    watchVariableRequest = _messages.MessageField('WatchVariableRequest', 2)