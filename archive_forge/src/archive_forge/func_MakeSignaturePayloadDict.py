from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from containerregistry.client import docker_name
from googlecloudsdk.core.exceptions import Error
import six
from six.moves import urllib
def MakeSignaturePayloadDict(container_image_url):
    """Creates a dict representing a JSON signature object to sign.

  Args:
    container_image_url: See `containerregistry.client.docker_name.Digest` for
      artifact URL validation and parsing details.  `container_image_url` must
      be a fully qualified image URL with a valid sha256 digest.

  Returns:
    Dictionary of nested dictionaries and strings, suitable for passing to
    `json.dumps` or similar.
  """
    url = ReplaceImageUrlScheme(image_url=container_image_url, scheme='')
    try:
        repo_digest = docker_name.Digest(url)
    except docker_name.BadNameException as e:
        raise BadImageUrlError(e)
    return {'critical': {'identity': {'docker-reference': six.text_type(repo_digest.as_repository())}, 'image': {'docker-manifest-digest': repo_digest.digest}, 'type': 'Google cloud binauthz container signature'}}