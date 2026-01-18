from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.osconfig.v1alpha1 import osconfig_v1alpha1_messages as messages
Test the access control policy for an OsConfig or an OS Config Assignment.

      Args:
        request: (OsconfigProjectsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      