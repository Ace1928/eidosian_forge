from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1DocumentationFile(_messages.Message):
    """Documentation file contents for a catalog item.

  Fields:
    contents: Required. The file contents. The max size is 4 MB.
    displayName: Required. A display name for the file, shown in the
      management UI. Max length is 255 characters.
  """
    contents = _messages.BytesField(1)
    displayName = _messages.StringField(2)