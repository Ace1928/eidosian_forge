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
def CreateTransferConfig(self, reference, data_source, target_dataset=None, display_name=None, refresh_window_days=None, params=None, auth_info=None, service_account_name=None, notification_pubsub_topic=None, schedule_args=None, destination_kms_key=None, location=None):
    """Create a transfer config corresponding to TransferConfigReference.

    Args:
      reference: the TransferConfigReference to create.
      data_source: The data source for the transfer config.
      target_dataset: The dataset where the new transfer config will exist.
      display_name: A display name for the transfer config.
      refresh_window_days: Refresh window days for the transfer config.
      params: Parameters for the created transfer config. The parameters should
        be in JSON format given as a string. Ex: --params="{'param':'value'}".
        The params should be the required values needed for each data source and
        will vary.
      auth_info: A dict contains authorization info which can be either an
        authorization_code or a version_info that the user input if they need
        credentials.
      service_account_name: The service account that the user could act as and
        used as the credential to create transfer runs from the transfer config.
      notification_pubsub_topic: The Pub/Sub topic where notifications will be
        sent after transfer runs associated with this transfer config finish.
      schedule_args: Optional parameters to customize data transfer schedule.
      destination_kms_key: Optional KMS key for encryption.
      location: The location where the new transfer config will run.

    Raises:
      BigqueryNotFoundError: if a requested item is not found.
      bq_error.BigqueryError: if a required field isn't provided.

    Returns:
      The generated transfer configuration name.
    """
    create_items = {}
    transfer_client = self.GetTransferV1ApiClient()
    if target_dataset:
        create_items['destinationDatasetId'] = target_dataset
    if display_name:
        create_items['displayName'] = display_name
    else:
        raise bq_error.BigqueryError('A display name must be provided.')
    create_items['dataSourceId'] = data_source
    if refresh_window_days:
        data_source_info = self._FetchDataSource(reference, data_source)
        create_items = bq_processor_utils.ProcessRefreshWindowDaysFlag(refresh_window_days, data_source_info, create_items, data_source)
    if params:
        create_items = bq_processor_utils.ProcessParamsFlag(params, create_items)
    else:
        raise bq_error.BigqueryError('Parameters must be provided.')
    if location:
        parent = reference + '/locations/' + location
    else:
        parent = reference + '/locations/-'
    if schedule_args:
        if schedule_args.schedule is not None:
            create_items['schedule'] = schedule_args.schedule
        create_items['scheduleOptions'] = schedule_args.ToScheduleOptionsPayload()
    if notification_pubsub_topic:
        create_items['notification_pubsub_topic'] = notification_pubsub_topic
    if destination_kms_key:
        create_items['encryption_configuration'] = {'kms_key_name': {'value': destination_kms_key}}
    new_transfer_config = transfer_client.projects().locations().transferConfigs().create(parent=parent, body=create_items, authorizationCode=None if auth_info is None else auth_info.get(AUTHORIZATION_CODE), versionInfo=None if auth_info is None else auth_info.get(VERSION_INFO), serviceAccountName=service_account_name).execute()
    return new_transfer_config['name']