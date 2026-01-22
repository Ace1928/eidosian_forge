from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AwsMetadata(_messages.Message):
    """AWS metadata associated with the resource, only applicable if the
  finding's cloud provider is Amazon Web Services.

  Fields:
    account: The AWS account associated with the resource.
    organization: The AWS organization associated with the resource.
    organizationalUnits: A list of AWS organizational units associated with
      the resource, ordered from lowest level (closest to the account) to
      highest level.
  """
    account = _messages.MessageField('AwsAccount', 1)
    organization = _messages.MessageField('AwsOrganization', 2)
    organizationalUnits = _messages.MessageField('AwsOrganizationalUnit', 3, repeated=True)