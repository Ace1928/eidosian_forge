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
def UpdateModel(self, reference, description=None, expiration=None, labels_to_set=None, label_keys_to_remove=None, vertex_ai_model_id=None, etag=None):
    """Updates a Model.

    Args:
      reference: the ModelReference to update.
      description: an optional description for model.
      expiration: optional expiration time in milliseconds since the epoch.
        Specifying 0 clears the expiration time for the model.
      labels_to_set: an optional dict of labels to set on this model.
      label_keys_to_remove: an optional list of label keys to remove from this
        model.
      vertex_ai_model_id: an optional string as Vertex AI model ID to register.
      etag: if set, checks that etag in the existing model matches.

    Raises:
      TypeError: if reference is not a ModelReference.
    """
    bq_id_utils.typecheck(reference, bq_id_utils.ApiClientHelper.ModelReference, method='UpdateModel')
    updated_model = {}
    if description is not None:
        updated_model['description'] = description
    if expiration is not None:
        updated_model['expirationTime'] = expiration or None
    if 'labels' not in updated_model:
        updated_model['labels'] = {}
    if labels_to_set:
        for label_key, label_value in labels_to_set.items():
            updated_model['labels'][label_key] = label_value
    if label_keys_to_remove:
        for label_key in label_keys_to_remove:
            updated_model['labels'][label_key] = None
    if vertex_ai_model_id is not None:
        updated_model['trainingRuns'] = [{'vertex_ai_model_id': vertex_ai_model_id}]
    request = self.GetModelsApiClient().models().patch(body=updated_model, projectId=reference.projectId, datasetId=reference.datasetId, modelId=reference.modelId)
    if etag:
        request.headers['If-Match'] = etag if etag else updated_model['etag']
    request.execute()