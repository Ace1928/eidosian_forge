from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Dict, NamedTuple, Optional
from clients import utils as bq_client_utils
from utils import bq_error
def MergeCapacityCommitments(client, project_id, location, capacity_commitment_ids):
    """Merges capacity commitments into one.

  Arguments:
    client: The client used to make the request.
    project_id: The project ID of the resources to update.
    location: Capacity commitments location.
    capacity_commitment_ids: List of capacity commitment ids.

  Returns:
    Merged capacity commitment.

  Raises:
    bq_error.BigqueryError: if capacity commitment cannot be merged.
  """
    if not project_id:
        raise bq_error.BigqueryError('project id must be specified.')
    if not location:
        raise bq_error.BigqueryError('location must be specified.')
    if capacity_commitment_ids is None or len(capacity_commitment_ids) < 2:
        raise bq_error.BigqueryError('at least 2 capacity commitments must be specified.')
    parent = 'projects/%s/locations/%s' % (project_id, location)
    body = {'capacityCommitmentIds': capacity_commitment_ids}
    return client.projects().locations().capacityCommitments().merge(parent=parent, body=body).execute()