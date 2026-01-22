from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
from googlecloudsdk.api_lib.util import exceptions as exceptions_util
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.core import exceptions
import six
class NoTLSError(exceptions.Error):
    """TLS 1.2 support is required to connect to GKE.

  Your Python installation does not support TLS 1.2. For Python2, please upgrade
  to version 2.7.9 or greater; for Python3, please upgrade to version 3.4 or
  greater.
  """