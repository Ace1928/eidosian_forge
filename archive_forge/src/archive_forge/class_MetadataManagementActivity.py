from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetadataManagementActivity(_messages.Message):
    """The metadata management activities of the metastore service.

  Fields:
    metadataExports: Output only. The latest metadata exports of the metastore
      service.
    restores: Output only. The latest restores of the metastore service.
  """
    metadataExports = _messages.MessageField('MetadataExport', 1, repeated=True)
    restores = _messages.MessageField('Restore', 2, repeated=True)