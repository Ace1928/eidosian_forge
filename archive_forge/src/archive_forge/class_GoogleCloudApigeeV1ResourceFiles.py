from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ResourceFiles(_messages.Message):
    """List of resource files.

  Fields:
    resourceFile: List of resource files.
  """
    resourceFile = _messages.MessageField('GoogleCloudApigeeV1ResourceFile', 1, repeated=True)