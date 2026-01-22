from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingOrganizationsUpdateCmekSettingsRequest(_messages.Message):
    """A LoggingOrganizationsUpdateCmekSettingsRequest object.

  Fields:
    cmekSettings: A CmekSettings resource to be passed as the request body.
    name: Required. The resource name for the CMEK settings to update.
      "projects/[PROJECT_ID]/cmekSettings"
      "organizations/[ORGANIZATION_ID]/cmekSettings"
      "billingAccounts/[BILLING_ACCOUNT_ID]/cmekSettings"
      "folders/[FOLDER_ID]/cmekSettings" For
      example:"organizations/12345/cmekSettings"Note: CMEK for the Log Router
      can currently only be configured for Google Cloud organizations. Once
      configured, it applies to all projects and folders in the Google Cloud
      organization.
    updateMask: Optional. Field mask identifying which fields from
      cmek_settings should be updated. A field will be overwritten if and only
      if it is in the update mask. Output only fields cannot be updated.See
      FieldMask for more information.For example: "updateMask=kmsKeyName"
  """
    cmekSettings = _messages.MessageField('CmekSettings', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)