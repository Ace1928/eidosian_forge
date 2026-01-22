from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeGlobalOrganizationOperationsDeleteRequest(_messages.Message):
    """A ComputeGlobalOrganizationOperationsDeleteRequest object.

  Fields:
    operation: Name of the Operations resource to delete.
    parentId: Parent ID for this request.
  """
    operation = _messages.StringField(1, required=True)
    parentId = _messages.StringField(2)