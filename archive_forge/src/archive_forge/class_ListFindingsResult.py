from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListFindingsResult(_messages.Message):
    """Result containing the Finding.

  Fields:
    finding: Finding matching the search request.
    resource: Output only. Resource that is associated with this finding.
  """
    finding = _messages.MessageField('GoogleCloudSecuritycenterV2Finding', 1)
    resource = _messages.MessageField('Resource', 2)