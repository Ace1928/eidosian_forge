from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class DimensionsValue(_messages.Message):
    """If this map is nonempty, then this override applies only to specific
    values for dimensions defined in the limit unit.  For example, an override
    on a limit with the unit 1/{project}/{region} could contain an entry with
    the key "region" and the value "us-east-1"; the override is only applied
    to quota consumed in that region.  This map has the following
    restrictions: - Keys that are not defined in the limit's unit are not
    valid keys.   Any string appearing in {brackets} in the unit (besides
    {project} or   {user}) is a defined key. - "project" is not a valid key;
    the project is already specified in   the parent resource name. - "user"
    is not a valid key; the API does not support quota overrides   that apply
    only to a specific user. - If "region" appears as a key, its value must be
    a valid Cloud region. - If "zone" appears as a key, its value must be a
    valid Cloud zone. - If any valid key other than "region" or "zone" appears
    in the map, then   all valid keys other than "region" or "zone" must also
    appear in the map.

    Messages:
      AdditionalProperty: An additional property for a DimensionsValue object.

    Fields:
      additionalProperties: Additional properties of type DimensionsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a DimensionsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)