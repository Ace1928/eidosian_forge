from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datacatalog.v1beta1 import datacatalog_v1beta1_messages as messages
Returns the permissions that a caller has on the specified taxonomy or policy tag.

      Args:
        request: (DatacatalogProjectsLocationsTaxonomiesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      