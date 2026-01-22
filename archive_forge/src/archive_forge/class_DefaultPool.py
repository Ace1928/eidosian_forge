from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DefaultPool(_messages.Message):
    """Execution using the default Cloud Build pool.

  Fields:
    artifactStorage: Optional. Cloud Storage location where execution outputs
      should be stored. This can either be a bucket ("gs://my-bucket") or a
      path within a bucket ("gs://my-bucket/my-dir"). If unspecified, a
      default bucket located in the same region will be used.
    serviceAccount: Optional. Google service account to use for execution. If
      unspecified, the project execution service account
      (-compute@developer.gserviceaccount.com) will be used.
  """
    artifactStorage = _messages.StringField(1)
    serviceAccount = _messages.StringField(2)