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
def _ParseDockerImage(img_str, err_msg, strict=True):
    """Validates and parses an image string into a DockerImage.

  Args:
    img_str: str, User input docker formatted string.
    err_msg: str, Error message to return to user.
    strict: bool, If False, defaults tags to "latest".

  Raises:
    ar_exceptions.InvalidInputValueError if user input is invalid.
    ar_exceptions.UnsupportedLocationError if provided location is invalid.

  Returns:
    A DockerImage, and a DockerTag or a DockerVersion.
  """
    try:
        docker_repo = _ParseInput(img_str)
    except ar_exceptions.InvalidInputValueError:
        raise ar_exceptions.InvalidInputValueError(_INVALID_DOCKER_IMAGE_ERROR)
    img_by_digest_match = re.match(DOCKER_IMG_BY_DIGEST_REGEX, img_str)
    if img_by_digest_match:
        docker_img = DockerImage(docker_repo, img_by_digest_match.group('img'))
        return (docker_img, DockerVersion(docker_img, img_by_digest_match.group('digest')))
    img_by_tag_match = re.match(DOCKER_IMG_BY_TAG_REGEX, img_str)
    if img_by_tag_match:
        docker_img = DockerImage(docker_repo, img_by_tag_match.group('img'))
        return (docker_img, DockerTag(docker_img, img_by_tag_match.group('tag')))
    whole_img_match = re.match(DOCKER_IMG_REGEX, img_str)
    if whole_img_match:
        docker_img = DockerImage(docker_repo, whole_img_match.group('img').strip('/'))
        return (docker_img, None if strict else DockerTag(docker_img, 'latest'))
    raise ar_exceptions.InvalidInputValueError(err_msg)