from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from contextlib import contextmanager
import re
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2 import docker_http as v2_docker_http
from containerregistry.client.v2 import docker_image as v2_image
from containerregistry.client.v2_2 import docker_http as v2_2_docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
from containerregistry.client.v2_2 import docker_image_list
from googlecloudsdk.api_lib.container.images import container_analysis_data_util
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import transports
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.docker import constants
from googlecloudsdk.core.docker import docker
from googlecloudsdk.core.util import times
import six
from six.moves import map
import six.moves.http_client
def GetDigestFromName(image_name):
    """Gets a digest object given a repository, tag or digest.

  Args:
    image_name: A docker image reference, possibly underqualified.

  Returns:
    a docker_name.Digest object.

  Raises:
    InvalidImageNameError: If no digest can be resolved.
  """
    tag_or_digest = GetDockerImageFromTagOrDigest(image_name)

    def ResolveV2Tag(tag):
        with v2_image.FromRegistry(basic_creds=CredentialProvider(), name=tag, transport=Http()) as v2_img:
            if v2_img.exists():
                return v2_img.digest()
            return None

    def ResolveV22Tag(tag):
        with v2_2_image.FromRegistry(basic_creds=CredentialProvider(), name=tag, transport=Http(), accepted_mimes=v2_2_docker_http.SUPPORTED_MANIFEST_MIMES) as v2_2_img:
            if v2_2_img.exists():
                return v2_2_img.digest()
            return None

    def ResolveManifestListTag(tag):
        with docker_image_list.FromRegistry(basic_creds=CredentialProvider(), name=tag, transport=Http()) as manifest_list:
            if manifest_list.exists():
                return manifest_list.digest()
            return None
    sha256 = ResolveManifestListTag(tag_or_digest) or ResolveV22Tag(tag_or_digest) or ResolveV2Tag(tag_or_digest)
    if not sha256:
        raise InvalidImageNameError('[{0}] is not found or is not a valid name. Expected tag in the form "base:tag" or "tag" or digest in the form "sha256:<digest>"'.format(image_name))
    if not isinstance(tag_or_digest, docker_name.Digest):
        log.warning('Successfully resolved tag to sha256, but it is recommended to use sha256 directly.')
    return docker_name.Digest('{registry}/{repository}@{sha256}'.format(registry=tag_or_digest.registry, repository=tag_or_digest.repository, sha256=sha256))