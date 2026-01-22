from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Dict, NamedTuple, Optional
from clients import utils as bq_client_utils
from utils import bq_error
def ListBiReservations(client, reference):
    """List BI reservations in the project and location for the given reference.

  Arguments:
    client: The client used to make the request.
    reference: Reservation reference containing project and location.

  Returns:
    List of BI reservations in the given project/location.
  """
    parent = 'projects/%s/locations/%s/biReservation' % (reference.projectId, reference.location)
    response = client.projects().locations().getBiReservation(name=parent).execute()
    return response