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
def _GetDefaultResources():
    """Gets default config values for project, location, and repository."""
    project = properties.VALUES.core.project.Get()
    location = properties.VALUES.artifacts.location.Get()
    repo = properties.VALUES.artifacts.repository.Get()
    if not project or not location or (not repo):
        raise ar_exceptions.InvalidInputValueError(_INVALID_DEFAULT_DOCKER_STRING_ERROR.format(**{'project': project, 'location': location, 'repo': repo}))
    return DockerRepo(project, location, repo)