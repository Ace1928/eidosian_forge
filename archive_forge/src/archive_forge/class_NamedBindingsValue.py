from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class NamedBindingsValue(_messages.Message):
    """For each non-reserved named binding site in the query string, there
    must be a named parameter with that name, but not necessarily the inverse.
    Key must match regex `A-Za-z_$*`, must not match regex `__.*__`, and must
    not be `""`.

    Messages:
      AdditionalProperty: An additional property for a NamedBindingsValue
        object.

    Fields:
      additionalProperties: Additional properties of type NamedBindingsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a NamedBindingsValue object.

      Fields:
        key: Name of the additional property.
        value: A GqlQueryParameter attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GqlQueryParameter', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)