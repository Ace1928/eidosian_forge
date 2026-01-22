from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
import struct
import sys
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import http_proxy_types
import httplib2
import six
from six.moves.urllib import parse
import socks
def _ExtractUnsignedInt16(binary_data):
    if len(binary_data) < 2:
        raise IncompleteData()
    return (struct.unpack(str('>H'), binary_data[:2])[0], binary_data[2:])