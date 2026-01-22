from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudcommerceconsumerprocurementProjectsEntitlementsGetRequest(_messages.Message):
    """A CloudcommerceconsumerprocurementProjectsEntitlementsGetRequest object.

  Fields:
    name: Required. The name of the entitlement to retrieve. This field is one
      of the following forms: `projects/{project-
      number}/entitlements/{entitlement-id}` `projects/{project-
      id}/entitlements/{entitlement-id}`.
  """
    name = _messages.StringField(1, required=True)