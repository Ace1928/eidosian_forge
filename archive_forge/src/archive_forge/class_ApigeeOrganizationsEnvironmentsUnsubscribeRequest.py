from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsUnsubscribeRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsUnsubscribeRequest object.

  Fields:
    googleCloudApigeeV1Subscription: A GoogleCloudApigeeV1Subscription
      resource to be passed as the request body.
    parent: Required. Name of the environment. Use the following structure in
      your request: `organizations/{org}/environments/{env}`
  """
    googleCloudApigeeV1Subscription = _messages.MessageField('GoogleCloudApigeeV1Subscription', 1)
    parent = _messages.StringField(2, required=True)