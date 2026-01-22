from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AWSV4Signature(_messages.Message):
    """The configuration needed to generate an AWS V4 Signature

  Fields:
    accessKeyId: Required. The access key ID that your origin uses to identify
      the key.
    originRegion: Required. The name of the AWS region that your origin is in.
    secretAccessKeyVersion: Required. The Secret Manager secret version of the
      secret access key used by your origin. This is the resource name of the
      secret version in the format `projects/*/secrets/*/versions/*` where the
      `*` values are replaced by the project, the secret, and the version that
      you require.
  """
    accessKeyId = _messages.StringField(1)
    originRegion = _messages.StringField(2)
    secretAccessKeyVersion = _messages.StringField(3)