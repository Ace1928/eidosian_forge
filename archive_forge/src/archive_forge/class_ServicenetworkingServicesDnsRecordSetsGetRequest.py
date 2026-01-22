from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicenetworkingServicesDnsRecordSetsGetRequest(_messages.Message):
    """A ServicenetworkingServicesDnsRecordSetsGetRequest object.

  Fields:
    consumerNetwork: Required. The consumer network containing the record set.
      Must be in the form of projects/{project}/global/networks/{network}
    domain: Required. The domain name of the zone containing the recordset.
    parent: Required. Parent resource identifying the connection which owns
      this collection of DNS zones in the format services/{service}.
    type: Required. RecordSet Type eg. type='A'. See the list of [Supported
      DNS Types](https://cloud.google.com/dns/records/json-record).
    zone: Required. The name of the zone containing the record set.
  """
    consumerNetwork = _messages.StringField(1)
    domain = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    type = _messages.StringField(4)
    zone = _messages.StringField(5)