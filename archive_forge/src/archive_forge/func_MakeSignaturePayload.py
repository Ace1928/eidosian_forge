from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from containerregistry.client import docker_name
from googlecloudsdk.core.exceptions import Error
import six
from six.moves import urllib
def MakeSignaturePayload(container_image_url):
    """Creates a JSON bytestring representing a signature object to sign.

  Args:
    container_image_url: See `containerregistry.client.docker_name.Digest` for
      artifact URL validation and parsing details.  `container_image_url` must
      be a fully qualified image URL with a valid sha256 digest.

  Returns:
    A bytestring representing a JSON-encoded structure of nested dictionaries
    and strings.
  """
    payload_dict = MakeSignaturePayloadDict(container_image_url)
    payload = json.dumps(payload_dict, ensure_ascii=True, indent=2, separators=(',', ': '), sort_keys=True)
    return '{}\n'.format(payload).encode('utf-8')