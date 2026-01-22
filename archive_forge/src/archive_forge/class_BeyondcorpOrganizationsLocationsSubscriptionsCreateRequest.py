from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BeyondcorpOrganizationsLocationsSubscriptionsCreateRequest(_messages.Message):
    """A BeyondcorpOrganizationsLocationsSubscriptionsCreateRequest object.

  Fields:
    googleCloudBeyondcorpSaasplatformSubscriptionsV1alphaSubscription: A
      GoogleCloudBeyondcorpSaasplatformSubscriptionsV1alphaSubscription
      resource to be passed as the request body.
    parent: Required. The resource name of the subscription location using the
      form: `organizations/{organization_id}/locations/{location}`
  """
    googleCloudBeyondcorpSaasplatformSubscriptionsV1alphaSubscription = _messages.MessageField('GoogleCloudBeyondcorpSaasplatformSubscriptionsV1alphaSubscription', 1)
    parent = _messages.StringField(2, required=True)