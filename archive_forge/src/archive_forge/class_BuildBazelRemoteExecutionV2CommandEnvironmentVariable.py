from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildBazelRemoteExecutionV2CommandEnvironmentVariable(_messages.Message):
    """An `EnvironmentVariable` is one variable to set in the running program's
  environment.

  Fields:
    name: The variable name.
    value: The variable value.
  """
    name = _messages.StringField(1)
    value = _messages.StringField(2)