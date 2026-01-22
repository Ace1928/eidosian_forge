from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsMulticastDomainsGetRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsMulticastDomainsGetRequest object.

  Fields:
    name: Required. The resource name of the multicast domain to get. Use the
      following format: `projects/*/locations/global/multicastDomains/*`.
  """
    name = _messages.StringField(1, required=True)