from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Dict, NamedTuple, Optional
from clients import utils as bq_client_utils
from utils import bq_error
def CreateReservation(client, api_version: str, reference, slots: int, ignore_idle_slots: bool, edition, target_job_concurrency: Optional[int], multi_region_auxiliary: Optional[bool], autoscale_max_slots: Optional[int]=None) -> Dict[str, Any]:
    """Create a reservation with the given reservation reference.

  Arguments:
    client: The client used to make the request.
    api_version: The api version to make the request against.
    reference: Reservation to create.
    slots: Number of slots allocated to this reservation subtree.
    ignore_idle_slots: Specifies whether queries should ignore idle slots from
      other reservations.
    edition: The edition for this reservation.
    target_job_concurrency: Job concurrency target.
    multi_region_auxiliary: Whether this reservation is for the auxiliary
      region.
    autoscale_max_slots: Number of slots to be scaled when needed.

  Returns:
    Reservation object that was created.

  Raises:
    bq_error.BigqueryError: if autoscale_max_slots is used with other
      version.
  """
    reservation = GetBodyForCreateReservation(api_version, slots, ignore_idle_slots, edition, target_job_concurrency, multi_region_auxiliary, autoscale_max_slots)
    parent = 'projects/%s/locations/%s' % (reference.projectId, reference.location)
    return client.projects().locations().reservations().create(parent=parent, body=reservation, reservationId=reference.reservationId).execute()