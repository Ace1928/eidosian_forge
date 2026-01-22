from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SourceProjectsReposListRequest(_messages.Message):
    """A SourceProjectsReposListRequest object.

  Fields:
    projectId: The project ID whose repos should be listed.
  """
    projectId = _messages.StringField(1, required=True)