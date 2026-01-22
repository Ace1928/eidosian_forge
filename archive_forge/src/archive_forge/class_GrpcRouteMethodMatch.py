from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GrpcRouteMethodMatch(_messages.Message):
    """Specifies a match against a method.

  Enums:
    TypeValueValuesEnum: Optional. Specifies how to match against the name. If
      not specified, a default value of "EXACT" is used.

  Fields:
    caseSensitive: Optional. Specifies that matches are case sensitive. The
      default value is true. case_sensitive must not be used with a type of
      REGULAR_EXPRESSION.
    grpcMethod: Required. Name of the method to match against. If unspecified,
      will match all methods.
    grpcService: Required. Name of the service to match against. If
      unspecified, will match all services.
    type: Optional. Specifies how to match against the name. If not specified,
      a default value of "EXACT" is used.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Optional. Specifies how to match against the name. If not specified, a
    default value of "EXACT" is used.

    Values:
      TYPE_UNSPECIFIED: Unspecified.
      EXACT: Will only match the exact name provided.
      REGULAR_EXPRESSION: Will interpret grpc_method and grpc_service as
        regexes. RE2 syntax is supported.
    """
        TYPE_UNSPECIFIED = 0
        EXACT = 1
        REGULAR_EXPRESSION = 2
    caseSensitive = _messages.BooleanField(1)
    grpcMethod = _messages.StringField(2)
    grpcService = _messages.StringField(3)
    type = _messages.EnumField('TypeValueValuesEnum', 4)