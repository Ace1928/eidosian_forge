from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V1Beta1QuotaOverride(_messages.Message):
    """A quota override

  Messages:
    DimensionsValue:  If this map is nonempty, then this override applies only
      to specific values for dimensions defined in the limit unit.  For
      example, an override on a limit with the unit 1/{project}/{region} could
      contain an entry with the key "region" and the value "us-east-1"; the
      override is only applied to quota consumed in that region.  This map has
      the following restrictions:  *   Keys that are not defined in the
      limit's unit are not valid keys.     Any string appearing in {brackets}
      in the unit (besides {project} or     {user}) is a defined key. *
      "project" is not a valid key; the project is already specified in
      the parent resource name. *   "user" is not a valid key; the API does
      not support quota overrides     that apply only to a specific user. *
      If "region" appears as a key, its value must be a valid Cloud region. *
      If "zone" appears as a key, its value must be a valid Cloud zone. *   If
      any valid key other than "region" or "zone" appears in the map, then
      all valid keys other than "region" or "zone" must also appear in the
      map.

  Fields:
    dimensions:  If this map is nonempty, then this override applies only to
      specific values for dimensions defined in the limit unit.  For example,
      an override on a limit with the unit 1/{project}/{region} could contain
      an entry with the key "region" and the value "us-east-1"; the override
      is only applied to quota consumed in that region.  This map has the
      following restrictions:  *   Keys that are not defined in the limit's
      unit are not valid keys.     Any string appearing in {brackets} in the
      unit (besides {project} or     {user}) is a defined key. *   "project"
      is not a valid key; the project is already specified in     the parent
      resource name. *   "user" is not a valid key; the API does not support
      quota overrides     that apply only to a specific user. *   If "region"
      appears as a key, its value must be a valid Cloud region. *   If "zone"
      appears as a key, its value must be a valid Cloud zone. *   If any valid
      key other than "region" or "zone" appears in the map, then     all valid
      keys other than "region" or "zone" must also appear in the     map.
    metric: The name of the metric to which this override applies.  An example
      name would be: `compute.googleapis.com/cpus`
    name: The resource name of the producer override. An example name would
      be: `services/compute.googleapis.com/projects/123/consumerQuotaMetrics/c
      ompute.googleapis.com%2Fcpus/limits/%2Fproject%2Fregion/producerOverride
      s/4a3f2c1d`
    overrideValue: The overriding quota limit value. Can be any nonnegative
      integer, or -1 (unlimited quota).
    unit: The limit unit of the limit to which this override applies.  An
      example unit would be: `1/{project}/{region}` Note that `{project}` and
      `{region}` are not placeholders in this example; the literal characters
      `{` and `}` occur in the string.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DimensionsValue(_messages.Message):
        """ If this map is nonempty, then this override applies only to specific
    values for dimensions defined in the limit unit.  For example, an override
    on a limit with the unit 1/{project}/{region} could contain an entry with
    the key "region" and the value "us-east-1"; the override is only applied
    to quota consumed in that region.  This map has the following
    restrictions:  *   Keys that are not defined in the limit's unit are not
    valid keys.     Any string appearing in {brackets} in the unit (besides
    {project} or     {user}) is a defined key. *   "project" is not a valid
    key; the project is already specified in     the parent resource name. *
    "user" is not a valid key; the API does not support quota overrides
    that apply only to a specific user. *   If "region" appears as a key, its
    value must be a valid Cloud region. *   If "zone" appears as a key, its
    value must be a valid Cloud zone. *   If any valid key other than "region"
    or "zone" appears in the map, then     all valid keys other than "region"
    or "zone" must also appear in the     map.

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
    dimensions = _messages.MessageField('DimensionsValue', 1)
    metric = _messages.StringField(2)
    name = _messages.StringField(3)
    overrideValue = _messages.IntegerField(4)
    unit = _messages.StringField(5)