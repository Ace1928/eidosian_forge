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
def _RecordSetCopy(record_set, api_version='v1'):
    """Returns a copy of the given record-set.

  Args:
    record_set: ResourceRecordSet, Record-set to be copied.
    api_version: [str], the api version to use for creating the records.

  Returns:
    Returns a copy of the given record-set.
  """
    messages = core_apis.GetMessagesModule('dns', api_version)
    copy = messages.ResourceRecordSet()
    copy.kind = record_set.kind
    copy.name = record_set.name
    copy.type = record_set.type
    copy.ttl = record_set.ttl
    copy.rrdatas = list(record_set.rrdatas)
    return copy