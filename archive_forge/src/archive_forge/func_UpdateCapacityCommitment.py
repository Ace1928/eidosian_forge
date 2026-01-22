from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Dict, NamedTuple, Optional
from clients import utils as bq_client_utils
from utils import bq_error
def UpdateCapacityCommitment(client, reference, plan, renewal_plan):
    """Updates a capacity commitment with the given reference.

  Arguments:
    client: The client used to make the request.
    reference: Capacity commitment to update.
    plan: Commitment plan for this capacity commitment.
    renewal_plan: Renewal plan for this capacity commitment.

  Returns:
    Capacity commitment object that was updated.

  Raises:
    bq_error.BigqueryError: if capacity commitment cannot be updated.
  """
    if plan is None and renewal_plan is None:
        raise bq_error.BigqueryError('Please specify fields to be updated.')
    capacity_commitment = {}
    update_mask = []
    if plan is not None:
        capacity_commitment['plan'] = plan
        update_mask.append('plan')
    if renewal_plan is not None:
        capacity_commitment['renewal_plan'] = renewal_plan
        update_mask.append('renewal_plan')
    return client.projects().locations().capacityCommitments().patch(name=reference.path(), updateMask=','.join(update_mask), body=capacity_commitment).execute()