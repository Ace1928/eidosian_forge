from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import re
from typing import Any, List, NamedTuple, Optional
from utils import bq_error
from utils import bq_id_utils
def MakeAzureFederatedAppClientIdPropertiesJson(federated_app_client_id: str) -> str:
    """Returns properties for a connection with a federated app (client) id.

  Args:
    federated_app_client_id: federated application (client) id.

  Returns:
    JSON string with properties to create a connection with customer's federated
    application (client) id.
  """
    return '{"federatedApplicationClientId": "%s"}' % federated_app_client_id