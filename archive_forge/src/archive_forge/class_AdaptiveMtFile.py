from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AdaptiveMtFile(_messages.Message):
    """An AdaptiveMtFile.

  Fields:
    createTime: Output only. Timestamp when this file was created.
    displayName: The file's display name.
    entryCount: The number of entries that the file contains.
    name: Required. The resource name of the file, in form of
      `projects/{project-number-or-id}/locations/{location_id}/adaptiveMtDatas
      ets/{dataset}/adaptiveMtFiles/{file}`
    updateTime: Output only. Timestamp when this file was last updated.
  """
    createTime = _messages.StringField(1)
    displayName = _messages.StringField(2)
    entryCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    name = _messages.StringField(4)
    updateTime = _messages.StringField(5)