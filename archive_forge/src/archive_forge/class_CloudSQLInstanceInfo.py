from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudSQLInstanceInfo(_messages.Message):
    """For display only. Metadata associated with a Cloud SQL instance.

  Fields:
    displayName: Name of a Cloud SQL instance.
    externalIp: External IP address of a Cloud SQL instance.
    internalIp: Internal IP address of a Cloud SQL instance.
    networkUri: URI of a Cloud SQL instance network or empty string if the
      instance does not have one.
    region: Region in which the Cloud SQL instance is running.
    uri: URI of a Cloud SQL instance.
  """
    displayName = _messages.StringField(1)
    externalIp = _messages.StringField(2)
    internalIp = _messages.StringField(3)
    networkUri = _messages.StringField(4)
    region = _messages.StringField(5)
    uri = _messages.StringField(6)