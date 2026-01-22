from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Dict, NamedTuple, Optional
from clients import utils as bq_client_utils
from utils import bq_error
def GetReservation(client, reference):
    """Gets a reservation with the given reservation reference.

  Arguments:
    client: The client used to make the request.
    reference: Reservation to get.

  Returns:
    Reservation object corresponding to the given id.
  """
    return client.projects().locations().reservations().get(name=reference.path()).execute()