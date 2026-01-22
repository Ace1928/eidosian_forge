from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class LocalizedLabelTextValue(_messages.Message):
    """Optional. A map of `labelText` to localize into other languages.
    Format the language in [ISO
    639](https://wikipedia.org/wiki/ISO_639_macrolanguage) and the
    country/region in [ISO 3166](https://wikipedia.org/wiki/ISO_3166),
    separated by a hyphen `-`. For example, `en-US`. If a user's locale is
    present in the map's keys, the user sees the localized version of the
    `labelText`.

    Messages:
      AdditionalProperty: An additional property for a LocalizedLabelTextValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        LocalizedLabelTextValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a LocalizedLabelTextValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)