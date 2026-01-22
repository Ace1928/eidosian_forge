from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Dict, NamedTuple, Optional
from clients import utils as bq_client_utils
from utils import bq_error
def MoveReservationAssignment(client, id_fallbacks: NamedTuple('IDS', [('project_id', Optional[str]), ('api_version', Optional[str])]), reference, destination_reservation_id, default_location):
    """Moves given reservation assignment under another reservation."""
    destination_reservation_reference = bq_client_utils.GetReservationReference(id_fallbacks=id_fallbacks, identifier=destination_reservation_id, default_location=default_location, check_reservation_project=False)
    body = {'destinationId': destination_reservation_reference.path()}
    return client.projects().locations().reservations().assignments().move(name=reference.path(), body=body).execute()