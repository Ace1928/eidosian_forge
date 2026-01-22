from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Dict, NamedTuple, Optional
from clients import utils as bq_client_utils
from utils import bq_error
def GetParamsForUpdateReservation(api_version: str, slots, ignore_idle_slots, target_job_concurrency: Optional[int], autoscale_max_slots):
    """Return the request body and update mask for UpdateReservation.

  Arguments:
    api_version: The api version to make the request against.
    slots: Number of slots allocated to this reservation subtree.
    ignore_idle_slots: Specifies whether queries should ignore idle slots from
      other reservations.
    target_job_concurrency: Job concurrency target.
    autoscale_max_slots: Number of slots to be scaled when needed.

  Returns:
    Reservation object that was updated.

  Raises:
    bq_error.BigqueryError: if autoscale_max_slots is used with other
      version.
  """
    reservation = {}
    update_mask = ''
    if slots is not None:
        reservation['slot_capacity'] = slots
        update_mask += 'slot_capacity,'
    if ignore_idle_slots is not None:
        reservation['ignore_idle_slots'] = ignore_idle_slots
        update_mask += 'ignore_idle_slots,'
    if target_job_concurrency is not None:
        reservation['concurrency'] = target_job_concurrency
        update_mask += 'concurrency,'
    if autoscale_max_slots is not None:
        if autoscale_max_slots != 0:
            reservation['autoscale'] = {}
            reservation['autoscale']['max_slots'] = autoscale_max_slots
            update_mask += 'autoscale.max_slots,'
        else:
            update_mask += 'autoscale,'
    return (reservation, update_mask)