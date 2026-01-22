from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding as api_encoding
from dns import rdatatype
from dns import zone
from googlecloudsdk.api_lib.dns import record_types
from googlecloudsdk.api_lib.dns import svcb_stub
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import encoding
import six
def _RecordSetFromZoneRecord(name, rdset, origin, api_version='v1'):
    """Returns the Cloud DNS ResourceRecordSet for the given zone file record.

  Args:
    name: Name, Domain name of the zone record.
    rdset: Rdataset, The zone record object.
    origin: Name, The origin domain of the zone file.
    api_version: [str], the api version to use for creating the records.

  Returns:
    The ResourceRecordSet equivalent for the given zone record, or None for
    unsupported record types.
  """
    if rdset.rdtype not in record_types.SUPPORTED_TYPES:
        return None
    messages = core_apis.GetMessagesModule('dns', api_version)
    record_set = messages.ResourceRecordSet()
    record_set.kind = record_set.kind
    record_set.name = name.derelativize(origin).to_text()
    record_set.ttl = rdset.ttl
    record_set.type = rdatatype.to_text(rdset.rdtype)
    rdatas = []
    for rdata in rdset:
        rdatas.append(GetRdataTranslation(rdset.rdtype)(rdata, origin))
    record_set.rrdatas = rdatas
    return record_set