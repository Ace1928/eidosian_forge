from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccesscontextmanagerAccessPoliciesAccessLevelsListRequest(_messages.Message):
    """A AccesscontextmanagerAccessPoliciesAccessLevelsListRequest object.

  Enums:
    AccessLevelFormatValueValuesEnum: Whether to return `BasicLevels` in the
      Cloud Common Expression language, as `CustomLevels`, rather than as
      `BasicLevels`. Defaults to returning `AccessLevels` in the format they
      were defined.

  Fields:
    accessLevelFormat: Whether to return `BasicLevels` in the Cloud Common
      Expression language, as `CustomLevels`, rather than as `BasicLevels`.
      Defaults to returning `AccessLevels` in the format they were defined.
    pageSize: Number of Access Levels to include in the list. Default 100.
    pageToken: Next page token for the next batch of Access Level instances.
      Defaults to the first page of results.
    parent: Required. Resource name for the access policy to list Access
      Levels from. Format: `accessPolicies/{policy_id}`
  """

    class AccessLevelFormatValueValuesEnum(_messages.Enum):
        """Whether to return `BasicLevels` in the Cloud Common Expression
    language, as `CustomLevels`, rather than as `BasicLevels`. Defaults to
    returning `AccessLevels` in the format they were defined.

    Values:
      LEVEL_FORMAT_UNSPECIFIED: The format was not specified.
      AS_DEFINED: Uses the format the resource was defined in. BasicLevels are
        returned as BasicLevels, CustomLevels are returned as CustomLevels.
      CEL: Use Cloud Common Expression Language when returning the resource.
        Both BasicLevels and CustomLevels are returned as CustomLevels.
    """
        LEVEL_FORMAT_UNSPECIFIED = 0
        AS_DEFINED = 1
        CEL = 2
    accessLevelFormat = _messages.EnumField('AccessLevelFormatValueValuesEnum', 1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)