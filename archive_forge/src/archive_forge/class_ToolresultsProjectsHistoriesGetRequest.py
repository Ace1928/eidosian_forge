from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ToolresultsProjectsHistoriesGetRequest(_messages.Message):
    """A ToolresultsProjectsHistoriesGetRequest object.

  Fields:
    historyId: A History id. Required.
    projectId: A Project id. Required.
  """
    historyId = _messages.StringField(1, required=True)
    projectId = _messages.StringField(2, required=True)