from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicenetworkingServicesDnsRecordSetsAddRequest(_messages.Message):
    """A ServicenetworkingServicesDnsRecordSetsAddRequest object.

  Fields:
    addDnsRecordSetRequest: A AddDnsRecordSetRequest resource to be passed as
      the request body.
    parent: Required. The service that is managing peering connectivity for a
      service producer's organization. For Google services that support this
      functionality, this value is
      `services/servicenetworking.googleapis.com`.
  """
    addDnsRecordSetRequest = _messages.MessageField('AddDnsRecordSetRequest', 1)
    parent = _messages.StringField(2, required=True)