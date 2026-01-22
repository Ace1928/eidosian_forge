from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AwsAccessKey(_messages.Message):
    """AWS access key (see [AWS Security
  Credentials](https://docs.aws.amazon.com/general/latest/gr/aws-security-
  credentials.html)). For information on our data retention policy for user
  credentials, see [User credentials](/storage-transfer/docs/data-
  retention#user-credentials).

  Fields:
    accessKeyId: Required. AWS access key ID.
    secretAccessKey: Required. AWS secret access key. This field is not
      returned in RPC responses.
  """
    accessKeyId = _messages.StringField(1)
    secretAccessKey = _messages.StringField(2)