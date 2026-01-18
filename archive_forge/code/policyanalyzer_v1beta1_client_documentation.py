from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.policyanalyzer.v1beta1 import policyanalyzer_v1beta1_messages as messages
Queries policy activities on GCP resources.

      Args:
        request: (PolicyanalyzerProjectsLocationsActivityTypesActivitiesQueryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudPolicyanalyzerV1beta1QueryActivityResponse) The response message.
      