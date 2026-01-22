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
def QuotedText(text):
    """Returns the given text within quotes.

  Args:
    text: str, The text to be escaped.

  Returns:
    str, The given text within quotes. For further details on why this is
    necessary, please look at the TXT section at:
    https://cloud.google.com/dns/what-is-cloud-dns#supported_record_types.
  """
    text = encoding.Decode(text)
    if text.startswith('"') and text.endswith('"'):
        return text
    else:
        return '"{0}"'.format(text)