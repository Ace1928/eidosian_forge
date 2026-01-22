from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datastore.v1 import datastore_v1_messages as messages
Queries for entities.

      Args:
        request: (DatastoreProjectsRunQueryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RunQueryResponse) The response message.
      