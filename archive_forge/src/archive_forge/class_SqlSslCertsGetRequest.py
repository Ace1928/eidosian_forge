from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlSslCertsGetRequest(_messages.Message):
    """A SqlSslCertsGetRequest object.

  Fields:
    instance: Cloud SQL instance ID. This does not include the project ID.
    project: Project ID of the project that contains the instance.
    sha1Fingerprint: Sha1 FingerPrint.
  """
    instance = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    sha1Fingerprint = _messages.StringField(3, required=True)