from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Dict, NamedTuple, Optional
from clients import utils as bq_client_utils
from utils import bq_error
def GetBodyForCreateReservation(api_version: str, slots: int, ignore_idle_slots: bool, edition, target_job_concurrency: Optional[int], multi_region_auxiliary: Optional[bool], autoscale_max_slots: Optional[int]=None) -> Dict[str, Any]:
    """Return the request body for CreateReservation.

  Arguments:
    api_version: The api version to make the request against.
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
    reservation = {}
    reservation['slot_capacity'] = slots
    reservation['ignore_idle_slots'] = ignore_idle_slots
    if multi_region_auxiliary is not None:
        reservation['multi_region_auxiliary'] = multi_region_auxiliary
    if target_job_concurrency is not None:
        reservation['concurrency'] = target_job_concurrency
    if autoscale_max_slots is not None:
        reservation['autoscale'] = {}
        reservation['autoscale']['max_slots'] = autoscale_max_slots
    if edition is not None:
        reservation['edition'] = edition
    return reservation