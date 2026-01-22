from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.admin.v1 import admin_v1_messages as messages
Returns the current set of valid backup verification codes for the specified user.

      Args:
        request: (DirectoryVerificationCodesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (VerificationCodes) The response message.
      