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
def CreateDataset(self, reference, ignore_existing=False, description=None, display_name=None, acl=None, default_table_expiration_ms=None, default_partition_expiration_ms=None, data_location=None, labels=None, default_kms_key=None, source_dataset_reference=None, external_source=None, connection_id=None, max_time_travel_hours=None, storage_billing_model=None, resource_tags=None):
    """Create a dataset corresponding to DatasetReference.

    Args:
      reference: the DatasetReference to create.
      ignore_existing: (boolean, default False) If False, raise an exception if
        the dataset already exists.
      description: an optional dataset description.
      display_name: an optional friendly name for the dataset.
      acl: an optional ACL for the dataset, as a list of dicts.
      default_table_expiration_ms: Default expiration time to apply to new
        tables in this dataset.
      default_partition_expiration_ms: Default partition expiration time to
        apply to new partitioned tables in this dataset.
      data_location: Location where the data in this dataset should be stored.
        Must be either 'EU' or 'US'. If specified, the project that owns the
        dataset must be enabled for data location.
      labels: An optional dict of labels.
      default_kms_key: An optional kms dey that will apply to all newly created
        tables in the dataset, if no explicit key is supplied in the creating
        request.
      source_dataset_reference: An optional ApiClientHelper.DatasetReference
        that will be the source of this linked dataset. #
      external_source: External source that backs this dataset.
      connection_id: Connection used for accessing the external_source.
      max_time_travel_hours: Optional. Define the max time travel in hours. The
        value can be from 48 to 168 hours (2 to 7 days). The default value is
        168 hours if this is not set.
      storage_billing_model: Optional. Sets the storage billing model for the
        dataset.
      resource_tags: an optional dict of tags to attach to the dataset.

    Raises:
      TypeError: if reference is not an ApiClientHelper.DatasetReference
        or if source_dataset_reference is provided but is not an
        bq_id_utils.ApiClientHelper.DatasetReference.
        or if both external_dataset_reference and source_dataset_reference
        are provided or if not all required arguments for external database is
        provided.
      BigqueryDuplicateError: if reference exists and ignore_existing
         is False.
    """
    return client_dataset.CreateDataset(apiclient=self.apiclient, reference=reference, ignore_existing=ignore_existing, description=description, display_name=display_name, acl=acl, default_table_expiration_ms=default_table_expiration_ms, default_partition_expiration_ms=default_partition_expiration_ms, data_location=data_location, labels=labels, default_kms_key=default_kms_key, source_dataset_reference=source_dataset_reference, external_source=external_source, connection_id=connection_id, max_time_travel_hours=max_time_travel_hours, storage_billing_model=storage_billing_model, resource_tags=resource_tags)