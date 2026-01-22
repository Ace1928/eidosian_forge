from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GrpcRouteHeaderModifier(_messages.Message):
    """Specifies how to modify gRPC headers in a request or a response.

  Messages:
    AddValue: Add the headers with given map where key is the name of the
      header, value is the value of the header.
    SetValue: Completely overwrite/replace the headers with given map where
      key is the name of the header, value is the value of the header.

  Fields:
    add: Add the headers with given map where key is the name of the header,
      value is the value of the header.
    remove: Remove headers (matching by header names) specified in the list.
    set: Completely overwrite/replace the headers with given map where key is
      the name of the header, value is the value of the header.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AddValue(_messages.Message):
        """Add the headers with given map where key is the name of the header,
    value is the value of the header.

    Messages:
      AdditionalProperty: An additional property for a AddValue object.

    Fields:
      additionalProperties: Additional properties of type AddValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AddValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class SetValue(_messages.Message):
        """Completely overwrite/replace the headers with given map where key is
    the name of the header, value is the value of the header.

    Messages:
      AdditionalProperty: An additional property for a SetValue object.

    Fields:
      additionalProperties: Additional properties of type SetValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a SetValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    add = _messages.MessageField('AddValue', 1)
    remove = _messages.StringField(2, repeated=True)
    set = _messages.MessageField('SetValue', 3)