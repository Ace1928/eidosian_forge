from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class InstancesImportRequest(_messages.Message):
    """Database instance import request.

  Fields:
    importContext: Contains details about the import operation.
  """
    importContext = _messages.MessageField('ImportContext', 1)