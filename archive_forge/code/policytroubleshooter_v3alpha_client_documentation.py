from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.policytroubleshooter.v3alpha import policytroubleshooter_v3alpha_messages as messages
Checks why an access is granted or not with service perimeters.

      Args:
        request: (GoogleCloudPolicytroubleshooterServiceperimeterV3alphaTroubleshootServicePerimeterRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudPolicytroubleshooterServiceperimeterV3alphaTroubleshootServicePerimeterResponse) The response message.
      