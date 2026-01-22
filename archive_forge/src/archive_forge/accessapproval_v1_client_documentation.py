from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.accessapproval.v1 import accessapproval_v1_messages as messages
Updates the settings associated with a project, folder, or organization. Settings to update are determined by the value of field_mask.

      Args:
        request: (AccessapprovalProjectsUpdateAccessApprovalSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AccessApprovalSettings) The response message.
      