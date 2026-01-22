from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConvertConversionWorkspaceRequest(_messages.Message):
    """Request message for 'ConvertConversionWorkspace' request.

  Fields:
    autoCommit: Optional. Specifies whether the conversion workspace is to be
      committed automatically after the conversion.
    convertFullPath: Optional. Automatically convert the full entity path for
      each entity specified by the filter. For example, if the filter
      specifies a table, that table schema (and database if there is one) will
      also be converted.
    filter: Optional. Filter the entities to convert. Leaving this field empty
      will convert all of the entities. Supports Google AIP-160 style
      filtering.
  """
    autoCommit = _messages.BooleanField(1)
    convertFullPath = _messages.BooleanField(2)
    filter = _messages.StringField(3)