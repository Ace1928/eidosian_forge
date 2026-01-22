from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSharedflowsListRequest(_messages.Message):
    """A ApigeeOrganizationsSharedflowsListRequest object.

  Fields:
    includeMetaData: Indicates whether to include shared flow metadata in the
      response.
    includeRevisions: Indicates whether to include a list of revisions in the
      response.
    parent: Required. The name of the parent organization under which to get
      shared flows. Must be of the form: `organizations/{organization_id}`
  """
    includeMetaData = _messages.BooleanField(1)
    includeRevisions = _messages.BooleanField(2)
    parent = _messages.StringField(3, required=True)