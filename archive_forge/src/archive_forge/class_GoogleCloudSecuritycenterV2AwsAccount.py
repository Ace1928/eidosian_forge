from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2AwsAccount(_messages.Message):
    """An AWS account that is a member of an organization.

  Fields:
    id: The unique identifier (ID) of the account, containing exactly 12
      digits.
    name: The friendly name of this account.
  """
    id = _messages.StringField(1)
    name = _messages.StringField(2)