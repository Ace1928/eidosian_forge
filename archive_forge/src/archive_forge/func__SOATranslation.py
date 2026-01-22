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
def _SOATranslation(rdata, origin):
    """Returns the translation of the given SOA rdata.

  Args:
    rdata: Rdata, The data to be translated.
    origin: Name, The origin domain name.

  Returns:
    str, The translation of the given SOA rdata which includes all the required
    SOA fields. Note that the primary NS name is left in a substitutable form
    because it is always provided by Cloud DNS.
  """
    return ' '.join((six.text_type(x) for x in ['{0}', rdata.rname.derelativize(origin), rdata.serial, rdata.refresh, rdata.retry, rdata.expire, rdata.minimum]))