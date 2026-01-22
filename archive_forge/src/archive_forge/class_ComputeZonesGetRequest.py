from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeZonesGetRequest(_messages.Message):
    """A ComputeZonesGetRequest object.

  Fields:
    project: Project ID for this request.
    zone: Name of the zone resource to return.
  """
    project = _messages.StringField(1, required=True)
    zone = _messages.StringField(2, required=True)