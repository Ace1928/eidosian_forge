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
class DockerImage(object):
    """Holder for a Docker image resource.

  A valid image has the format of
  LOCATION-docker.DOMAIN/PROJECT-ID/REPOSITORY-ID/IMAGE_PATH

  Properties:
    project: str, The name of cloud project.
    docker_repo: DockerRepo, The Docker repository.
    pkg: str, The name of the package.
  """

    def __init__(self, docker_repo, pkg_id):
        self._docker_repo = docker_repo
        self._pkg = pkg_id

    @property
    def project(self):
        return self._docker_repo.project

    @property
    def docker_repo(self):
        return self._docker_repo

    @property
    def pkg(self):
        return self._pkg

    def __eq__(self, other):
        if isinstance(other, DockerImage):
            return self._docker_repo == other._docker_repo and self._pkg == other._pkg
        return NotImplemented

    def GetPackageName(self):
        return '{}/packages/{}'.format(self.docker_repo.GetRepositoryName(), self.pkg.replace('/', '%2F'))

    def GetDockerString(self):
        return '{}{}-docker.{}/{}/{}/{}'.format(properties.VALUES.artifacts.registry_endpoint_prefix.Get(), self.docker_repo.location, properties.VALUES.artifacts.domain.Get(), self.docker_repo.project, self.docker_repo.repo, self.pkg.replace('%2F', '/'))