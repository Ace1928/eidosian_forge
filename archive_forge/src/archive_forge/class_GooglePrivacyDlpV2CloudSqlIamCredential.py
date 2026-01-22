from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2CloudSqlIamCredential(_messages.Message):
    """Use IAM auth to connect. This requires the Cloud SQL IAM feature to be
  enabled on the instance, which is not the default for Cloud SQL. See
  https://cloud.google.com/sql/docs/postgres/authentication and
  https://cloud.google.com/sql/docs/mysql/authentication.
  """