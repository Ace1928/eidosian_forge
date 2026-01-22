from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecretManagerSource(_messages.Message):
    """Configuration for secrets stored in Google Secret Manager.

  Fields:
    secretVersion: Required. The name of the Secret Version containing the
      encryption key in the following format:
      `projects/{project}/secrets/{secret_id}/versions/{version_number}` Note
      that only numbered versions are supported. Aliases like "latest" are not
      supported.
  """
    secretVersion = _messages.StringField(1)