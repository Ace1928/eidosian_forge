from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Dict, NamedTuple, Optional
from clients import utils as bq_client_utils
from utils import bq_error
def ListReservations(client, reference, page_size, page_token):
    """List reservations in the project and location for the given reference.

  Arguments:
    client: The client used to make the request.
    reference: Reservation reference containing project and location.
    page_size: Number of results to show.
    page_token: Token to retrieve the next page of results.

  Returns:
    Reservation object that was created.
  """
    parent = 'projects/%s/locations/%s' % (reference.projectId, reference.location)
    return client.projects().locations().reservations().list(parent=parent, pageSize=page_size, pageToken=page_token).execute()