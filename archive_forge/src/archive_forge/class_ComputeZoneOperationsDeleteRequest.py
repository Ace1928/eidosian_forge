from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeZoneOperationsDeleteRequest(_messages.Message):
    """A ComputeZoneOperationsDeleteRequest object.

  Fields:
    operation: Name of the Operations resource to delete.
    project: Project ID for this request.
    zone: Name of the zone for this request.
  """
    operation = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    zone = _messages.StringField(3, required=True)