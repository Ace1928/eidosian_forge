from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExportPolicy(_messages.Message):
    """Defines the export policy for the volume.

  Fields:
    rules: Required. List of export policy rules
  """
    rules = _messages.MessageField('SimpleExportPolicyRule', 1, repeated=True)