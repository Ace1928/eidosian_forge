from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional
import uuid
from absl import flags
from google.api_core.iam import Policy
from googleapiclient import http as http_request
import inflection
from clients import bigquery_client
from clients import client_dataset
from clients import client_reservation
from clients import table_reader as bq_table_reader
from clients import utils as bq_client_utils
from utils import bq_api_utils
from utils import bq_error
from utils import bq_id_utils
from utils import bq_processor_utils
def GetRowAccessPolicyIAMPolicy(self, reference: 'bq_id_utils.ApiClientHelper.RowAccessPolicyReference') -> Policy:
    """Gets IAM policy for the given row access policy resource.

    Arguments:
      reference: the RowAccessPolicyReference for the row access policy
        resource.

    Returns:
      The IAM policy attached to the given row access policy resource.

    Raises:
      TypeError: if reference is not a RowAccessPolicyReference.
    """
    bq_id_utils.typecheck(reference, bq_id_utils.ApiClientHelper.RowAccessPolicyReference, method='GetRowAccessPolicyIAMPolicy')
    formatted_resource = 'projects/%s/datasets/%s/tables/%s/rowAccessPolicies/%s' % (reference.projectId, reference.datasetId, reference.tableId, reference.policyId)
    return self.GetIAMPolicyApiClient().rowAccessPolicies().getIamPolicy(resource=formatted_resource).execute()