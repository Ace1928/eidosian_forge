from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingExclusionsCreateRequest(_messages.Message):
    """A LoggingExclusionsCreateRequest object.

  Fields:
    logExclusion: A LogExclusion resource to be passed as the request body.
    parent: Required. The parent resource in which to create the exclusion:
      "projects/[PROJECT_ID]" "organizations/[ORGANIZATION_ID]"
      "billingAccounts/[BILLING_ACCOUNT_ID]" "folders/[FOLDER_ID]" For
      examples:"projects/my-logging-project" "organizations/123456789"
  """
    logExclusion = _messages.MessageField('LogExclusion', 1)
    parent = _messages.StringField(2, required=True)