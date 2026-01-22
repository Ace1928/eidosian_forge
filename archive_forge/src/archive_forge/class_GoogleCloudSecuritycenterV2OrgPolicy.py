from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2OrgPolicy(_messages.Message):
    """Contains information about the org policies associated with the finding.

  Fields:
    name: The resource name of the org policy. Example:
      "organizations/{organization_id}/policies/{constraint_name}"
  """
    name = _messages.StringField(1)