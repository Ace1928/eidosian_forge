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
def _GetRDataReplacement(rdtype):
    """Gets the RData replacement function for this type.

  Args:
    rdtype: RDataType, the type for which to fetch a replacement function.

  Returns:
    A function for replacing rdata of a record-set with rdata from another
    record-set with the same name and type.
  """
    if rdtype == rdatatype.SOA:
        return _SOAReplacement
    return _RDataReplacement