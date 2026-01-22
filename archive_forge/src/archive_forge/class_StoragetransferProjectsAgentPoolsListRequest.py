from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StoragetransferProjectsAgentPoolsListRequest(_messages.Message):
    """A StoragetransferProjectsAgentPoolsListRequest object.

  Fields:
    filter: An optional list of query parameters specified as JSON text in the
      form of: `{"agentPoolNames":["agentpool1","agentpool2",...]}` Since
      `agentPoolNames` support multiple values, its values must be specified
      with array notation. When the filter is either empty or not provided,
      the list returns all agent pools for the project.
    pageSize: The list page size. The max allowed value is `256`.
    pageToken: The list page token.
    projectId: Required. The ID of the Google Cloud project that owns the job.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    projectId = _messages.StringField(4, required=True)