from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1SystemTimestamps(_messages.Message):
    """Timestamps associated with this resource in a particular system.

  Fields:
    createTime: Creation timestamp of the resource within the given system.
    expireTime: Output only. Expiration timestamp of the resource within the
      given system. Currently only applicable to BigQuery resources.
    updateTime: Timestamp of the last modification of the resource or its
      metadata within a given system. Note: Depending on the source system,
      not every modification updates this timestamp. For example, BigQuery
      timestamps every metadata modification but not data or permission
      changes.
  """
    createTime = _messages.StringField(1)
    expireTime = _messages.StringField(2)
    updateTime = _messages.StringField(3)