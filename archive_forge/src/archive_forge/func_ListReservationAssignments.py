from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Dict, NamedTuple, Optional
from clients import utils as bq_client_utils
from utils import bq_error
def ListReservationAssignments(client, reference, page_size, page_token):
    """Lists reservation assignments for given project and location.

  Arguments:
    client: The client used to make the request.
    reference: Reservation reference for the parent.
    page_size: Number of results to show.
    page_token: Token to retrieve the next page of results.

  Returns:
    ReservationAssignment object that was created.
  """
    return client.projects().locations().reservations().assignments().list(parent=reference.path(), pageSize=page_size, pageToken=page_token).execute()