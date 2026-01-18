from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.policyanalyzer.v1 import policyanalyzer_v1_messages as messages
Queries policy activities on Google Cloud resources.

      Args:
        request: (PolicyanalyzerProjectsLocationsActivityTypesActivitiesQueryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudPolicyanalyzerV1QueryActivityResponse) The response message.
      