from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IngestAssetResponse(_messages.Message):
    """Response of ingesting an other-cloud asset.

  Fields:
    name: It is the original name of the resource. For AWS assets, use
      [ARN](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference-
      arns.html)
  """
    name = _messages.StringField(1)