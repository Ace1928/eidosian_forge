from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from typing import Generator
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha import gkehub_v1alpha_messages as messages
def UpdateFleet(self, req: messages.GkehubProjectsLocationsFleetsPatchRequest) -> messages.Operation:
    """Updates a fleet resource from the Fleet API.

    Args:
      req: An HTTP patch request to send to the API server.

    Returns:
      A long-running operation to be polled till completion, or returned
      directly if user specifies async flag.

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
    return self.client.projects_locations_fleets.Patch(req)