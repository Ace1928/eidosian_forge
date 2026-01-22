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
def UpdateTransferConfig(self, reference, target_dataset=None, display_name=None, refresh_window_days=None, params=None, auth_info=None, service_account_name=None, destination_kms_key=None, notification_pubsub_topic=None, schedule_args=None):
    """Updates a transfer config.

    Args:
      reference: the TransferConfigReference to update.
      target_dataset: Optional updated target dataset.
      display_name: Optional change to the display name.
      refresh_window_days: Optional update to the refresh window days. Some data
        sources do not support this.
      params: Optional parameters to update.
      auth_info: A dict contains authorization info which can be either an
        authorization_code or a version_info that the user input if they want to
        update credentials.
      service_account_name: The service account that the user could act as and
        used as the credential to create transfer runs from the transfer config.
      destination_kms_key: Optional KMS key for encryption.
      notification_pubsub_topic: The Pub/Sub topic where notifications will be
        sent after transfer runs associated with this transfer config finish.
      schedule_args: Optional parameters to customize data transfer schedule.

    Raises:
      TypeError: if reference is not a TransferConfigReference.
      BigqueryNotFoundError: if dataset is not found
      bq_error.BigqueryError: required field not given.
    """
    bq_id_utils.typecheck(reference, bq_id_utils.ApiClientHelper.TransferConfigReference, method='UpdateTransferConfig')
    project_reference = 'projects/' + bq_client_utils.GetProjectReference(id_fallbacks=self).projectId
    transfer_client = self.GetTransferV1ApiClient()
    current_config = transfer_client.projects().locations().transferConfigs().get(name=reference.transferConfigName).execute()
    update_mask = []
    update_items = {}
    update_items['dataSourceId'] = current_config['dataSourceId']
    if target_dataset:
        dataset_reference = bq_client_utils.GetDatasetReference(id_fallbacks=self, identifier=target_dataset)
        if self.DatasetExists(dataset_reference):
            update_items['destinationDatasetId'] = target_dataset
            update_mask.append('transfer_config.destination_dataset_id')
        else:
            raise bq_error.BigqueryNotFoundError('Unknown %r' % (dataset_reference,), {'reason': 'notFound'}, [])
        update_items['destinationDatasetId'] = target_dataset
    if display_name:
        update_mask.append('transfer_config.display_name')
        update_items['displayName'] = display_name
    if params:
        update_items = bq_processor_utils.ProcessParamsFlag(params, update_items)
        update_mask.append('transfer_config.params')
    if refresh_window_days:
        data_source_info = self._FetchDataSource(project_reference, current_config['dataSourceId'])
        update_items = bq_processor_utils.ProcessRefreshWindowDaysFlag(refresh_window_days, data_source_info, update_items, current_config['dataSourceId'])
        update_mask.append('transfer_config.data_refresh_window_days')
    if schedule_args:
        if schedule_args.schedule is not None:
            update_items['schedule'] = schedule_args.schedule
            update_mask.append('transfer_config.schedule')
        update_items['scheduleOptions'] = schedule_args.ToScheduleOptionsPayload(options_to_copy=current_config.get('scheduleOptions'))
        update_mask.append('transfer_config.scheduleOptions')
    if notification_pubsub_topic:
        update_items['notification_pubsub_topic'] = notification_pubsub_topic
        update_mask.append('transfer_config.notification_pubsub_topic')
    if auth_info is not None and AUTHORIZATION_CODE in auth_info:
        update_mask.append(AUTHORIZATION_CODE)
    if auth_info is not None and VERSION_INFO in auth_info:
        update_mask.append(VERSION_INFO)
    if service_account_name:
        update_mask.append('service_account_name')
    if destination_kms_key:
        update_items['encryption_configuration'] = {'kms_key_name': {'value': destination_kms_key}}
        update_mask.append('encryption_configuration.kms_key_name')
    transfer_client.projects().locations().transferConfigs().patch(body=update_items, name=reference.transferConfigName, updateMask=','.join(update_mask), authorizationCode=None if auth_info is None else auth_info.get(AUTHORIZATION_CODE), versionInfo=None if auth_info is None else auth_info.get(VERSION_INFO), serviceAccountName=service_account_name, x__xgafv='2').execute()