from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1RelatedAccountGroup(_messages.Message):
    """A group of related accounts.

  Fields:
    name: Required. Identifier. The resource name for the related account
      group in the format
      `projects/{project}/relatedaccountgroups/{related_account_group}`.
  """
    name = _messages.StringField(1)