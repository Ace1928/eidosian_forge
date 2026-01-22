from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import random
import string
import time
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py.exceptions import HttpError
from apitools.base.py.exceptions import HttpNotFoundError
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import logs as cb_logs
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.services import enable_api as services_api
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as http_exc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.cloudbuild import execution
from googlecloudsdk.command_lib.compute.sole_tenancy import util as sole_tenancy_util
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding as encoding_util
import six
def GetDefaultBuilder(executable, docker_image_tag):
    """Return Docker image path for GCR builder wrapper.

  Args:
    executable: name of builder executable to run
    docker_image_tag: tag for Docker builder images (e.g. 'release')

  Returns:
    str: path to Docker images for GCR builder.
  """
    gcp_project = GetGcpProjectName(executable)
    gcr_image_get_api_url = 'https://gcr.io/v2/{gcp_project}/{executable}/manifests/{tag}'
    fallback_project_name = _COMPUTE_IMAGE_TOOLS_PROJECT_NAME
    if IsGcrImageExist(gcr_image_get_api_url.format(gcp_project=gcp_project, executable=executable, tag=docker_image_tag)):
        return _DEFAULT_BUILDER_DOCKER_PATTERN.format(gcp_project=gcp_project, executable=executable, docker_image_tag=docker_image_tag)
    return _DEFAULT_BUILDER_DOCKER_PATTERN.format(gcp_project=fallback_project_name, executable=executable, docker_image_tag=docker_image_tag)