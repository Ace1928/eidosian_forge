from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from containerregistry.client import docker_name
from googlecloudsdk.core.exceptions import Error
import six
from six.moves import urllib
def ReplaceImageUrlScheme(image_url, scheme):
    """Returns the passed `image_url` with the scheme replaced.

  Args:
    image_url: The URL to replace (or strip) the scheme from. (string)
    scheme: The scheme of the returned URL.  If this is an empty string or
      `None`, the scheme is stripped and the leading `//` of the resulting URL
      will be stripped off.
  Raises:
    BadImageUrlError: `image_url` isn't valid.
  """
    scheme = scheme or ''
    parsed_url = urllib.parse.urlparse(image_url)
    if parsed_url.scheme and (not parsed_url.netloc):
        raise BadImageUrlError("Image URL '{image_url}' is invalid because it does not have a host component.".format(image_url=image_url))
    if not parsed_url.netloc:
        parsed_url = urllib.parse.urlparse('//{}'.format(image_url))
    return parsed_url._replace(scheme=scheme).geturl().lstrip('/')