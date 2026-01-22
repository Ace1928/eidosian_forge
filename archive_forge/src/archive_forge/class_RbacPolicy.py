from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RbacPolicy(_messages.Message):
    """A RbacPolicy object.

  Fields:
    name: Name of the RbacPolicy.
    permissions: The list of permissions.
    principals: The list of principals.
  """
    name = _messages.StringField(1)
    permissions = _messages.MessageField('Permission', 2, repeated=True)
    principals = _messages.MessageField('Principal', 3, repeated=True)