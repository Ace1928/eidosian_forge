from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsInstancesNatAddressesDeleteRequest(_messages.Message):
    """A ApigeeOrganizationsInstancesNatAddressesDeleteRequest object.

  Fields:
    name: Required. Name of the nat address. Use the following structure in
      your request:
      `organizations/{org}/instances/{instances}/natAddresses/{nataddress}``
  """
    name = _messages.StringField(1, required=True)