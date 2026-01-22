from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from containerregistry.client import docker_name
from googlecloudsdk.core.exceptions import Error
import six
from six.moves import urllib
def PaeEncode(dsse_type, body):
    """Pae encode input using the specified dsse type.

  Args:
    dsse_type: DSSE envelope type.
    body: payload string.

  Returns:
    Pae-encoded payload byte string.
  """
    dsse_type_bytes = dsse_type.encode('utf-8')
    body_bytes = body.encode('utf-8')
    return b' '.join([b'DSSEv1', b'%d' % len(dsse_type_bytes), dsse_type_bytes, b'%d' % len(body_bytes), body_bytes])