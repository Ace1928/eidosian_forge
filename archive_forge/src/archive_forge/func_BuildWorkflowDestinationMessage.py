from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.eventarc import common
from googlecloudsdk.api_lib.eventarc.base import EventarcClientBase
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.eventarc import types
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import iso_duration
from googlecloudsdk.core.util import times
def BuildWorkflowDestinationMessage(self, project_id, destination_workflow, destination_workflow_location):
    """Builds a Workflow Destination message with the given data.

    Args:
      project_id: the ID of the project.
      destination_workflow: str or None, the Trigger's destination Workflow ID.
      destination_workflow_location: str or None, the location of the Trigger's
        destination Workflow. It defaults to the Trigger's location.

    Returns:
      A Destination message with a Workflow destination.
    """
    workflow_message = 'projects/{}/locations/{}/workflows/{}'.format(project_id, destination_workflow_location, destination_workflow)
    return self._messages.Destination(workflow=workflow_message)