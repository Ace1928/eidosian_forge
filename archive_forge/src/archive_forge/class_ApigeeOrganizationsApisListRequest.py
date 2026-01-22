from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsApisListRequest(_messages.Message):
    """A ApigeeOrganizationsApisListRequest object.

  Fields:
    includeMetaData: Flag that specifies whether to include API proxy metadata
      in the response.
    includeRevisions: Flag that specifies whether to include a list of
      revisions in the response.
    parent: Required. Name of the organization in the following format:
      `organizations/{org}`
  """
    includeMetaData = _messages.BooleanField(1)
    includeRevisions = _messages.BooleanField(2)
    parent = _messages.StringField(3, required=True)