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
def _SOAReplacement(current_record, record_to_be_imported, api_version='v1'):
    """Returns the replacement SOA record with restored primary NS name.

  Args:
    current_record: ResourceRecordSet, Current record-set.
    record_to_be_imported: ResourceRecordSet, Record-set to be imported.
    api_version: [str], the api version to use for creating the records.

  Returns:
    ResourceRecordSet, the replacement SOA record with restored primary NS name.
  """
    replacement = _RecordSetCopy(record_to_be_imported, api_version=api_version)
    replacement.rrdatas[0] = replacement.rrdatas[0].format(current_record.rrdatas[0].split()[0])
    if replacement == current_record:
        return NextSOARecordSet(replacement, api_version)
    else:
        return replacement