from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.util import common_args
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.artifacts import containeranalysis_util as ca_util
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def AddDockerTag(args):
    """Adds a Docker tag."""
    src_image, version_or_tag = _ParseDockerImage(args.DOCKER_IMAGE, _INVALID_DOCKER_IMAGE_ERROR)
    if version_or_tag is None:
        raise ar_exceptions.InvalidInputValueError(_INVALID_DOCKER_IMAGE_ERROR)
    dest_image, tag = _ParseDockerTag(args.DOCKER_TAG)
    if src_image.GetPackageName() != dest_image.GetPackageName():
        raise ar_exceptions.InvalidInputValueError('Image {}\ndoes not match image {}'.format(src_image.GetDockerString(), dest_image.GetDockerString()))
    _ValidateDockerRepo(src_image.docker_repo.GetRepositoryName())
    client = ar_requests.GetClient()
    messages = ar_requests.GetMessages()
    docker_version = version_or_tag
    if isinstance(version_or_tag, DockerTag):
        docker_version = DockerVersion(version_or_tag.image, ar_requests.GetVersionFromTag(client, messages, version_or_tag.GetTagName()))
    try:
        ar_requests.GetTag(client, messages, tag.GetTagName())
    except api_exceptions.HttpNotFoundError:
        ar_requests.CreateDockerTag(client, messages, tag, docker_version)
    else:
        ar_requests.DeleteTag(client, messages, tag.GetTagName())
        ar_requests.CreateDockerTag(client, messages, tag, docker_version)
    log.status.Print('Added tag [{}] to image [{}].'.format(tag.GetDockerString(), args.DOCKER_IMAGE))