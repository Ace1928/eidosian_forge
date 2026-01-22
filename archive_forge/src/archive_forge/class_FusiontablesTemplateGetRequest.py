from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class FusiontablesTemplateGetRequest(_messages.Message):
    """A FusiontablesTemplateGetRequest object.

  Fields:
    tableId: Table to which the template belongs
    templateId: Identifier for the template that is being requested
  """
    tableId = _messages.StringField(1, required=True)
    templateId = _messages.IntegerField(2, required=True, variant=_messages.Variant.INT32)