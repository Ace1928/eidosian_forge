from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.oslogin.v1beta import oslogin_v1beta_messages as messages
Adds an SSH public key and returns the profile information. Default POSIX account information is set when no username and UID exist as part of the login profile.

      Args:
        request: (OsloginUsersImportSshPublicKeyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ImportSshPublicKeyResponse) The response message.
      