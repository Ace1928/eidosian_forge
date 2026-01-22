from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class GlobalSettingsValue(_messages.Message):
    """Optional. A generic list of settings for the workspace. The settings
    are database pair dependant and can indicate default behavior for the
    mapping rules engine or turn on or off specific features. Such examples
    can be: convert_foreign_key_to_interleave=true, skip_triggers=false,
    ignore_non_table_synonyms=true

    Messages:
      AdditionalProperty: An additional property for a GlobalSettingsValue
        object.

    Fields:
      additionalProperties: Additional properties of type GlobalSettingsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a GlobalSettingsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)