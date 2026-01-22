from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Dict, NamedTuple, Optional
from clients import utils as bq_client_utils
from utils import bq_error
def DeleteCapacityCommitment(client, reference, force=None):
    """Deletes a capacity commitment with the given capacity commitment reference.

  Arguments:
    client: The client used to make the request.
    reference: Capacity commitment to delete.
    force: Force delete capacity commitment, ignoring commitment end time.
  """
    client.projects().locations().capacityCommitments().delete(name=reference.path(), force=force).execute()