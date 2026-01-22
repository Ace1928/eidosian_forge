from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceID(_messages.Message):
    """ResourceID encapsulates the definition of the identity of a resource.

  Fields:
    name: Name is the name of the resource. This name must be unique within
      the type.
    type: Type is the name of the resource.
  """
    name = _messages.StringField(1)
    type = _messages.StringField(2)