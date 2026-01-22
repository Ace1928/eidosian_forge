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
def UpdateConnection(self, reference, connection_type, properties, connection_credential=None, display_name=None, description=None, kms_key_name=None, connector_configuration=None):
    """Update connection with the given connection reference.

    Arguments:
      reference: Connection to update
      connection_type: Type of connection, allowed values: ['CLOUD_SQL']
      properties: Connection properties
      connection_credential: Connection credentials in JSON format.
      display_name: Friendly name for the connection
      description: Description of the connection
      kms_key_name: Optional KMS key name.
      connector_configuration: Optional configuration for connector
    Raises:
      bq_error.BigqueryClientError: The connection type is not defined
        when updating
      connection_credential or properties.
    Returns:
      Connection object that was created.
    """
    if (connection_credential or properties) and (not connection_type):
        raise bq_error.BigqueryClientError('connection_type is required when updating connection_credential or properties')
    connection = {}
    update_mask = []

    def GetUpdateMask(base_path, json_properties):
        """Creates an update mask from json_properties.

      Arguments:
        base_path: 'cloud_sql'
        json_properties: { 'host': ... , 'instanceId': ... }

      Returns:
         list of  paths in snake case:
         mask = ['cloud_sql.host', 'cloud_sql.instance_id']
      """
        return [base_path + '.' + inflection.underscore(json_property) for json_property in json_properties]

    def GetUpdateMaskRecursively(prefix, json_value):
        if not isinstance(json_value, dict) or not json_value:
            return [inflection.underscore(prefix)]
        result = []
        for name in json_value:
            new_prefix = prefix + '.' + name
            new_json_value = json_value.get(name)
            result.extend(GetUpdateMaskRecursively(new_prefix, new_json_value))
        return result
    if display_name:
        connection['friendlyName'] = display_name
        update_mask.append('friendlyName')
    if description:
        connection['description'] = description
        update_mask.append('description')
    if kms_key_name is not None:
        update_mask.append('kms_key_name')
    if kms_key_name:
        connection['kmsKeyName'] = kms_key_name
    if connection_type == 'CLOUD_SQL':
        if properties:
            cloudsql_properties = bq_processor_utils.ParseJson(properties)
            connection['cloudSql'] = cloudsql_properties
            update_mask.extend(GetUpdateMask(connection_type.lower(), cloudsql_properties))
        else:
            connection['cloudSql'] = {}
        if connection_credential:
            connection['cloudSql']['credential'] = bq_processor_utils.ParseJson(connection_credential)
            update_mask.append('cloudSql.credential')
    elif connection_type == 'AWS':
        if properties:
            aws_properties = bq_processor_utils.ParseJson(properties)
            connection['aws'] = aws_properties
            if aws_properties.get('crossAccountRole') and aws_properties['crossAccountRole'].get('iamRoleId'):
                update_mask.append('aws.crossAccountRole.iamRoleId')
            if aws_properties.get('accessRole') and aws_properties['accessRole'].get('iamRoleId'):
                update_mask.append('aws.access_role.iam_role_id')
        else:
            connection['aws'] = {}
        if connection_credential:
            connection['aws']['credential'] = bq_processor_utils.ParseJson(connection_credential)
            update_mask.append('aws.credential')
    elif connection_type == 'Azure':
        if properties:
            azure_properties = bq_processor_utils.ParseJson(properties)
            connection['azure'] = azure_properties
            if azure_properties.get('customerTenantId'):
                update_mask.append('azure.customer_tenant_id')
            if azure_properties.get('federatedApplicationClientId'):
                update_mask.append('azure.federated_application_client_id')
    elif connection_type == 'SQL_DATA_SOURCE':
        if properties:
            sql_data_source_properties = bq_processor_utils.ParseJson(properties)
            connection['sqlDataSource'] = sql_data_source_properties
            update_mask.extend(GetUpdateMask(connection_type.lower(), sql_data_source_properties))
        else:
            connection['sqlDataSource'] = {}
        if connection_credential:
            connection['sqlDataSource']['credential'] = bq_processor_utils.ParseJson(connection_credential)
            update_mask.append('sqlDataSource.credential')
    elif connection_type == 'CLOUD_SPANNER':
        if properties:
            cloudspanner_properties = bq_processor_utils.ParseJson(properties)
            connection['cloudSpanner'] = cloudspanner_properties
            update_mask.extend(GetUpdateMask(connection_type.lower(), cloudspanner_properties))
        else:
            connection['cloudSpanner'] = {}
    elif connection_type == 'SPARK':
        if properties:
            spark_properties = bq_processor_utils.ParseJson(properties)
            connection['spark'] = spark_properties
            if 'sparkHistoryServerConfig' in spark_properties:
                update_mask.append('spark.spark_history_server_config')
            if 'metastoreServiceConfig' in spark_properties:
                update_mask.append('spark.metastore_service_config')
        else:
            connection['spark'] = {}
    elif connector_configuration:
        connection['configuration'] = bq_processor_utils.ParseJson(connector_configuration)
        update_mask.extend(GetUpdateMaskRecursively('configuration', connection['configuration']))
    client = self.GetConnectionV1ApiClient()
    return client.projects().locations().connections().patch(name=reference.path(), updateMask=','.join(update_mask), body=connection).execute()