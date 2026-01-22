from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildBazelRemoteExecutionV2ToolDetails(_messages.Message):
    """Details for the tool used to call the API.

  Fields:
    toolName: Name of the tool, e.g. bazel.
    toolVersion: Version of the tool used for the request, e.g. 5.0.3.
  """
    toolName = _messages.StringField(1)
    toolVersion = _messages.StringField(2)