from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisProjectsScanConfigsListRequest(_messages.Message):
    """A ContaineranalysisProjectsScanConfigsListRequest object.

  Fields:
    filter: The filter expression.
    pageSize: The number of items to return.
    pageToken: The page token to use for the next request.
    parent: This containers the project Id i.e.: projects/{project_id}
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)