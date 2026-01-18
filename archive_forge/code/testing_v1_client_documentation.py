from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.testing.v1 import testing_v1_messages as messages
Gets the catalog of supported test environments. May return any of the following canonical error codes: - INVALID_ARGUMENT - if the request is malformed - NOT_FOUND - if the environment type does not exist - INTERNAL - if an internal error occurred.

      Args:
        request: (TestingTestEnvironmentCatalogGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestEnvironmentCatalog) The response message.
      