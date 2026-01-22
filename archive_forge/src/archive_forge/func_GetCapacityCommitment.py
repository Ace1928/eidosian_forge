from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Dict, NamedTuple, Optional
from clients import utils as bq_client_utils
from utils import bq_error
def GetCapacityCommitment(client, reference):
    """Gets a capacity commitment with the given capacity commitment reference.

  Arguments:
    client: The client used to make the request.
    reference: Capacity commitment to get.

  Returns:
    Capacity commitment object corresponding to the given id.
  """
    return client.projects().locations().capacityCommitments().get(name=reference.path()).execute()