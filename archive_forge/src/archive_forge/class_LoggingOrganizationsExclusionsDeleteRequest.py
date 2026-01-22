from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingOrganizationsExclusionsDeleteRequest(_messages.Message):
    """A LoggingOrganizationsExclusionsDeleteRequest object.

  Fields:
    name: Required. The resource name of an existing exclusion to delete:
      "projects/[PROJECT_ID]/exclusions/[EXCLUSION_ID]"
      "organizations/[ORGANIZATION_ID]/exclusions/[EXCLUSION_ID]"
      "billingAccounts/[BILLING_ACCOUNT_ID]/exclusions/[EXCLUSION_ID]"
      "folders/[FOLDER_ID]/exclusions/[EXCLUSION_ID]" For
      example:"projects/my-project/exclusions/my-exclusion"
  """
    name = _messages.StringField(1, required=True)