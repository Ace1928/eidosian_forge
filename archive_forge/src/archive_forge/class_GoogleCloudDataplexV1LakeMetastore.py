from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1LakeMetastore(_messages.Message):
    """Settings to manage association of Dataproc Metastore with a lake.

  Fields:
    service: Optional. A relative reference to the Dataproc Metastore
      (https://cloud.google.com/dataproc-metastore/docs) service associated
      with the lake:
      projects/{project_id}/locations/{location_id}/services/{service_id}
  """
    service = _messages.StringField(1)