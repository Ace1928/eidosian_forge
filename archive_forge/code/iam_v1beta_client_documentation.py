from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v1beta import iam_v1beta_messages as messages
Undeletes a WorkloadIdentityPool, as long as it was deleted fewer than 30 days ago.

      Args:
        request: (IamProjectsLocationsWorkloadIdentityPoolsUndeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      