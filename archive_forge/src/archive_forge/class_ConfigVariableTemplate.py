from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigVariableTemplate(_messages.Message):
    """ConfigVariableTemplate provides metadata about a `ConfigVariable` that
  is used in a Connection.

  Enums:
    StateValueValuesEnum: State of the config variable.
    ValueTypeValueValuesEnum: Type of the parameter: string, int, bool etc.
      consider custom type for the benefit for the validation.

  Fields:
    authorizationCodeLink: Authorization code link options. To be populated if
      `ValueType` is `AUTHORIZATION_CODE`
    description: Description.
    displayName: Display name of the parameter.
    enumOptions: Enum options. To be populated if `ValueType` is `ENUM`
    isAdvanced: Indicates if current template is part of advanced settings
    key: Key of the config variable.
    required: Flag represents that this `ConfigVariable` must be provided for
      a connection.
    requiredCondition: Condition under which a field would be required. The
      condition can be represented in the form of a logical expression.
    roleGrant: Role grant configuration for the config variable.
    state: State of the config variable.
    validationRegex: Regular expression in RE2 syntax used for validating the
      `value` of a `ConfigVariable`.
    valueType: Type of the parameter: string, int, bool etc. consider custom
      type for the benefit for the validation.
  """

    class StateValueValuesEnum(_messages.Enum):
        """State of the config variable.

    Values:
      STATE_UNSPECIFIED: Status is unspecified.
      ACTIVE: Config variable is active
      DEPRECATED: Config variable is deprecated.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        DEPRECATED = 2

    class ValueTypeValueValuesEnum(_messages.Enum):
        """Type of the parameter: string, int, bool etc. consider custom type for
    the benefit for the validation.

    Values:
      VALUE_TYPE_UNSPECIFIED: Value type is not specified.
      STRING: Value type is string.
      INT: Value type is integer.
      BOOL: Value type is boolean.
      SECRET: Value type is secret.
      ENUM: Value type is enum.
      AUTHORIZATION_CODE: Value type is authorization code.
      ENCRYPTION_KEY: Encryption Key.
    """
        VALUE_TYPE_UNSPECIFIED = 0
        STRING = 1
        INT = 2
        BOOL = 3
        SECRET = 4
        ENUM = 5
        AUTHORIZATION_CODE = 6
        ENCRYPTION_KEY = 7
    authorizationCodeLink = _messages.MessageField('AuthorizationCodeLink', 1)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    enumOptions = _messages.MessageField('EnumOption', 4, repeated=True)
    isAdvanced = _messages.BooleanField(5)
    key = _messages.StringField(6)
    required = _messages.BooleanField(7)
    requiredCondition = _messages.MessageField('LogicalExpression', 8)
    roleGrant = _messages.MessageField('RoleGrant', 9)
    state = _messages.EnumField('StateValueValuesEnum', 10)
    validationRegex = _messages.StringField(11)
    valueType = _messages.EnumField('ValueTypeValueValuesEnum', 12)