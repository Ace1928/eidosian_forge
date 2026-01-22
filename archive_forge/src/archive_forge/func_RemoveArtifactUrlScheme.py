from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from containerregistry.client import docker_name
from googlecloudsdk.core.exceptions import Error
import six
from six.moves import urllib
def RemoveArtifactUrlScheme(artifact_url):
    """Ensures the given URL has no scheme (e.g.

  replaces "https://gcr.io/foo/bar" with "gcr.io/foo/bar" and leaves
  "gcr.io/foo/bar" unchanged).

  Args:
    artifact_url: A URL string.
  Returns:
    The URL with the scheme removed.
  """
    url_without_scheme = ReplaceImageUrlScheme(artifact_url, scheme='')
    try:
        docker_name.Digest(url_without_scheme)
    except docker_name.BadNameException as e:
        raise BadImageUrlError(e)
    return url_without_scheme