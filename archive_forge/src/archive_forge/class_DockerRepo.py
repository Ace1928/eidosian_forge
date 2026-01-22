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
class DockerRepo(object):
    """Holder for a Docker repository.

  A valid Docker repository has the format of
  LOCATION-docker.DOMAIN/PROJECT-ID/REPOSITORY-ID

  Properties:
    project: str, The name of cloud project.
    location: str, The location of the Docker resource.
    repo: str, The name of the repository.
  """

    def __init__(self, project_id, location_id, repo_id):
        self._project = project_id
        self._location = location_id
        self._repo = repo_id

    @property
    def project(self):
        return self._project

    @property
    def location(self):
        return self._location

    @property
    def repo(self):
        return self._repo

    def __eq__(self, other):
        if isinstance(other, DockerRepo):
            return self._project == other._project and self._location == other._location and (self._repo == other._repo)
        return NotImplemented

    def GetDockerString(self):
        return '{}-docker.{}/{}/{}'.format(self.location, properties.VALUES.artifacts.domain.Get(), self.project, self.repo)

    def GetRepositoryName(self):
        return 'projects/{}/locations/{}/repositories/{}'.format(self.project, self.location, self.repo)