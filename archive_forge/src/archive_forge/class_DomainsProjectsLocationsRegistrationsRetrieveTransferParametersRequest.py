from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DomainsProjectsLocationsRegistrationsRetrieveTransferParametersRequest(_messages.Message):
    """A DomainsProjectsLocationsRegistrationsRetrieveTransferParametersRequest
  object.

  Fields:
    domainName: Required. The domain name. Unicode domain names must be
      expressed in Punycode format.
    location: Required. The location. Must be in the format
      `projects/*/locations/*`.
  """
    domainName = _messages.StringField(1)
    location = _messages.StringField(2, required=True)