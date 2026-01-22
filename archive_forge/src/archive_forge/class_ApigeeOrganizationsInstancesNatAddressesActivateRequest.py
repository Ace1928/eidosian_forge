from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsInstancesNatAddressesActivateRequest(_messages.Message):
    """A ApigeeOrganizationsInstancesNatAddressesActivateRequest object.

  Fields:
    googleCloudApigeeV1ActivateNatAddressRequest: A
      GoogleCloudApigeeV1ActivateNatAddressRequest resource to be passed as
      the request body.
    name: Required. Name of the nat address. Use the following structure in
      your request:
      `organizations/{org}/instances/{instances}/natAddresses/{nataddress}``
  """
    googleCloudApigeeV1ActivateNatAddressRequest = _messages.MessageField('GoogleCloudApigeeV1ActivateNatAddressRequest', 1)
    name = _messages.StringField(2, required=True)