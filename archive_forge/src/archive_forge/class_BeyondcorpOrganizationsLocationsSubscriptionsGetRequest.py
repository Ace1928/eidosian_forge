from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BeyondcorpOrganizationsLocationsSubscriptionsGetRequest(_messages.Message):
    """A BeyondcorpOrganizationsLocationsSubscriptionsGetRequest object.

  Fields:
    name: Required. The resource name of Subscription using the form: `organiz
      ations/{organization_id}/locations/{location}/subscriptions/{subscriptio
      n_id}`
  """
    name = _messages.StringField(1, required=True)