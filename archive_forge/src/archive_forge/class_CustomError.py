from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomError(_messages.Message):
    """Customize service error responses.  For example, list any service
  specific protobuf types that can appear in error detail lists of error
  responses.  Example:      custom_error:       types:       -
  google.foo.v1.CustomError       - google.foo.v1.AnotherError

  Fields:
    rules: The list of custom error rules to select to which messages this
      should apply.
    types: The list of custom error detail types, e.g.
      'google.foo.v1.CustomError'.
  """
    rules = _messages.MessageField('CustomErrorRule', 1, repeated=True)
    types = _messages.StringField(2, repeated=True)