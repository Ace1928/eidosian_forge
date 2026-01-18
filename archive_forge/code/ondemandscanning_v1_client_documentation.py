from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.ondemandscanning.v1 import ondemandscanning_v1_messages as messages
Initiates an analysis of the provided packages.

      Args:
        request: (OndemandscanningProjectsLocationsScansAnalyzePackagesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      