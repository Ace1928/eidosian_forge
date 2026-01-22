from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2Volume(_messages.Message):
    """Volume represents a named volume in a container.

  Fields:
    cloudSqlInstance: For Cloud SQL volumes, contains the specific instances
      that should be mounted. Visit
      https://cloud.google.com/sql/docs/mysql/connect-run for more information
      on how to connect Cloud SQL and Cloud Run.
    emptyDir: Ephemeral storage used as a shared volume.
    gcs: Persistent storage backed by a Google Cloud Storage bucket.
    name: Required. Volume's name.
    nfs: For NFS Voumes, contains the path to the nfs Volume
    secret: Secret represents a secret that should populate this volume.
  """
    cloudSqlInstance = _messages.MessageField('GoogleCloudRunV2CloudSqlInstance', 1)
    emptyDir = _messages.MessageField('GoogleCloudRunV2EmptyDirVolumeSource', 2)
    gcs = _messages.MessageField('GoogleCloudRunV2GCSVolumeSource', 3)
    name = _messages.StringField(4)
    nfs = _messages.MessageField('GoogleCloudRunV2NFSVolumeSource', 5)
    secret = _messages.MessageField('GoogleCloudRunV2SecretVolumeSource', 6)