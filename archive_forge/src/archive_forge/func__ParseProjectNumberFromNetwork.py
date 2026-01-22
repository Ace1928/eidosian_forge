from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.api_lib.services import peering
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.projects import util as projects_command_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _ParseProjectNumberFromNetwork(network, user_project):
    """Retrieves the project field from the provided network value."""
    try:
        registry = resources.REGISTRY.Clone()
        network_ref = registry.Parse(network, collection='compute.networks')
        project_identifier = network_ref.project
    except resources.Error:
        project_identifier = user_project
    return projects_command_util.GetProjectNumber(project_identifier)