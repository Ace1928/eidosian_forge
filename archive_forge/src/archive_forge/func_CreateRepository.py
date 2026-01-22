from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from apitools.base.py import exceptions as base_exceptions
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.artifacts import requests
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def CreateRepository(repo, skip_activation_prompt=False):
    """Creates an Artifact Registry repostiory and waits for the operation.

  Args:
    repo: googlecloudsdk.command_lib.artifacts.docker_util.DockerRepo defining
      the repository to be created.
    skip_activation_prompt: True if
  """
    messages = requests.GetMessages()
    repository_message = messages.Repository(name=repo.GetRepositoryName(), description='Cloud Run Source Deployments', format=messages.Repository.FormatValueValuesEnum.DOCKER)
    op = requests.CreateRepository(repo.project, repo.location, repository_message, skip_activation_prompt)
    op_resource = resources.REGISTRY.ParseRelativeName(op.name, collection='artifactregistry.projects.locations.operations')
    client = requests.GetClient()
    waiter.WaitFor(waiter.CloudOperationPoller(client.projects_locations_repositories, client.projects_locations_operations), op_resource)