from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.api_lib.app import logs_util
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.docker import constants
from googlecloudsdk.core.docker import docker
from googlecloudsdk.third_party.appengine.api import appinfo
def ValidateImageUrl(image_url, services):
    """Check the user-provided image URL.

  Ensures that:
  - it is consistent with the services being deployed (there must be exactly
    one)
  - it is an image in a supported Docker registry

  Args:
    image_url: str, the URL of the image to deploy provided by the user
    services: list, the services to deploy

  Raises:
    MultiDeployError: if image_url is provided and more than one service is
      being deployed
    docker.UnsupportedRegistryError: if image_url is provided and does not point
      to one of the supported registries
  """
    if image_url is None:
        return
    if len(services) != 1:
        raise exceptions.MultiDeployError()
    for registry in constants.ALL_SUPPORTED_REGISTRIES:
        if image_url.startswith(registry):
            return
    raise docker.UnsupportedRegistryError(image_url)