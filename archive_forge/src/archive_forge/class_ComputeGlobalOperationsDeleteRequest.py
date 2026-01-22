from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeGlobalOperationsDeleteRequest(_messages.Message):
    """A ComputeGlobalOperationsDeleteRequest object.

  Fields:
    operation: Name of the Operations resource to delete.
    project: Project ID for this request.
  """
    operation = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)