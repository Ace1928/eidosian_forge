from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ControlPlaneAccess(_messages.Message):
    """ControlPlaneAccess is the request body and response body of
  UpdateControlPlaneAccess. and the response body of GetControlPlaneAccess.
  The input identities contains an array of service accounts to grant access
  to the respective control plane resource, with each service account
  specified using the following format: `serviceAccount:`***service-account-
  name***. The ***service-account-name*** is formatted like an email address.
  For example: `my-control-plane-
  service_account@my_project_id.iam.gserviceaccount.com` You might specify
  multiple service accounts, for example, if you have multiple environments
  and wish to assign a unique service account to each one.

  Fields:
    loggerIdentities: Array of service accounts to grant access to control
      plane resources (for the Logger component).
    name: The resource name of the ControlPlaneAccess. Format:
      "organizations/{org}/controlPlaneAccess"
    synchronizerIdentities: Required. Array of service accounts to grant
      access to control plane resources (for the Synchronizer component). The
      service accounts must have **Apigee Synchronizer Manager** role. See
      also [Create service
      accounts](https://cloud.google.com/apigee/docs/hybrid/latest/sa-
      about#create-the-service-accounts).
    udcaIdentities: Required. Array of service accounts to grant access to
      control plane resources (for the UDCA component).
  """
    loggerIdentities = _messages.StringField(1, repeated=True)
    name = _messages.StringField(2)
    synchronizerIdentities = _messages.StringField(3, repeated=True)
    udcaIdentities = _messages.StringField(4, repeated=True)