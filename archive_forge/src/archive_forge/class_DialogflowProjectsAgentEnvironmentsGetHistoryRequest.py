from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsAgentEnvironmentsGetHistoryRequest(_messages.Message):
    """A DialogflowProjectsAgentEnvironmentsGetHistoryRequest object.

  Fields:
    pageSize: Optional. The maximum number of items to return in a single
      page. By default 100 and at most 1000.
    pageToken: Optional. The next_page_token value returned from a previous
      list request.
    parent: Required. The name of the environment to retrieve history for.
      Supported formats: - `projects//agent/environments/` -
      `projects//locations//agent/environments/` The environment ID for the
      default environment is `-`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)