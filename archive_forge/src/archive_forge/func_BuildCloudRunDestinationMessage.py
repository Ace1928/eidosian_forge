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
def BuildCloudRunDestinationMessage(self, destination_run_service, destination_run_job, destination_run_path, destination_run_region):
    """Builds a Destination message for a destination Cloud Run service.

    Args:
      destination_run_service: str or None, the destination Cloud Run service.
      destination_run_job: this destination is not supported in the beta API,
        but is included as an argument here for method consistency with v1.
      destination_run_path: str or None, the path on the destination Cloud Run
        service.
      destination_run_region: str or None, the destination Cloud Run service's
        region.

    Returns:
      A Destination message for a destination Cloud Run service.
    """
    del destination_run_job
    run_message = self._messages.CloudRunService(service=destination_run_service, path=destination_run_path, region=destination_run_region)
    return self._messages.Destination(cloudRunService=run_message)