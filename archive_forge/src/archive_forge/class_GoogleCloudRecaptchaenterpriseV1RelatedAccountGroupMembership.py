from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1RelatedAccountGroupMembership(_messages.Message):
    """A membership in a group of related accounts.

  Fields:
    accountId: The unique stable account identifier of the member. The
      identifier corresponds to an `account_id` provided in a previous
      `CreateAssessment` or `AnnotateAssessment` call.
    hashedAccountId: Deprecated: use `account_id` instead. The unique stable
      hashed account identifier of the member. The identifier corresponds to a
      `hashed_account_id` provided in a previous `CreateAssessment` or
      `AnnotateAssessment` call.
    name: Required. Identifier. The resource name for this membership in the
      format `projects/{project}/relatedaccountgroups/{relatedaccountgroup}/me
      mberships/{membership}`.
  """
    accountId = _messages.StringField(1)
    hashedAccountId = _messages.BytesField(2)
    name = _messages.StringField(3)