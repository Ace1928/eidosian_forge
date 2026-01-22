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
def IsOnlySOAIncrement(change, api_version='v1'):
    """Returns True if the change only contains an SOA increment, False otherwise.

  Args:
    change: Change, the change to be checked
    api_version: [str], the api version to use for creating the records.

  Returns:
    True if the change only contains an SOA increment, False otherwise.
  """
    return len(change.additions) == len(change.deletions) == 1 and _ToStandardEnumTypeSafe(change.deletions[0].type) is rdatatype.SOA and (NextSOARecordSet(change.deletions[0], api_version) == change.additions[0])