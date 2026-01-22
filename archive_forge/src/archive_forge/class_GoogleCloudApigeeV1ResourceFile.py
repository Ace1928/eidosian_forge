from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ResourceFile(_messages.Message):
    """Metadata about a resource file.

  Fields:
    name: ID of the resource file.
    type: Resource file type. {{ resource_file_type }}
  """
    name = _messages.StringField(1)
    type = _messages.StringField(2)