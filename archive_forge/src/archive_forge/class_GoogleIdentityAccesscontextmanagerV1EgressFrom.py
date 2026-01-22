from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIdentityAccesscontextmanagerV1EgressFrom(_messages.Message):
    """Defines the conditions under which an EgressPolicy matches a request.
  Conditions based on information about the source of the request. Note that
  if the destination of the request is also protected by a ServicePerimeter,
  then that ServicePerimeter must have an IngressPolicy which allows access in
  order for this request to succeed.

  Enums:
    IdentityTypeValueValuesEnum: Specifies the type of identities that are
      allowed access to outside the perimeter. If left unspecified, then
      members of `identities` field will be allowed access.
    SourceRestrictionValueValuesEnum: Whether to enforce traffic restrictions
      based on `sources` field. If the `sources` fields is non-empty, then
      this field must be set to `SOURCE_RESTRICTION_ENABLED`.

  Fields:
    identities: A list of identities that are allowed access through
      [EgressPolicy]. Identities can be an individual user, service account,
      Google group, or third-party identity. The `v1` identities that have the
      prefix `user`, `group`, `serviceAccount`, `principal`, and
      `principalSet` in https://cloud.google.com/iam/docs/principal-
      identifiers#v1 are supported.
    identityType: Specifies the type of identities that are allowed access to
      outside the perimeter. If left unspecified, then members of `identities`
      field will be allowed access.
    sourceRestriction: Whether to enforce traffic restrictions based on
      `sources` field. If the `sources` fields is non-empty, then this field
      must be set to `SOURCE_RESTRICTION_ENABLED`.
    sources: Sources that this EgressPolicy authorizes access from. If this
      field is not empty, then `source_restriction` must be set to
      `SOURCE_RESTRICTION_ENABLED`.
  """

    class IdentityTypeValueValuesEnum(_messages.Enum):
        """Specifies the type of identities that are allowed access to outside
    the perimeter. If left unspecified, then members of `identities` field
    will be allowed access.

    Values:
      IDENTITY_TYPE_UNSPECIFIED: No blanket identity group specified.
      ANY_IDENTITY: Authorize access from all identities outside the
        perimeter.
      ANY_USER_ACCOUNT: Authorize access from all human users outside the
        perimeter.
      ANY_SERVICE_ACCOUNT: Authorize access from all service accounts outside
        the perimeter.
    """
        IDENTITY_TYPE_UNSPECIFIED = 0
        ANY_IDENTITY = 1
        ANY_USER_ACCOUNT = 2
        ANY_SERVICE_ACCOUNT = 3

    class SourceRestrictionValueValuesEnum(_messages.Enum):
        """Whether to enforce traffic restrictions based on `sources` field. If
    the `sources` fields is non-empty, then this field must be set to
    `SOURCE_RESTRICTION_ENABLED`.

    Values:
      SOURCE_RESTRICTION_UNSPECIFIED: Enforcement preference unspecified, will
        not enforce traffic restrictions based on `sources` in EgressFrom.
      SOURCE_RESTRICTION_ENABLED: Enforcement preference enabled, traffic
        restrictions will be enforced based on `sources` in EgressFrom.
      SOURCE_RESTRICTION_DISABLED: Enforcement preference disabled, will not
        enforce traffic restrictions based on `sources` in EgressFrom.
    """
        SOURCE_RESTRICTION_UNSPECIFIED = 0
        SOURCE_RESTRICTION_ENABLED = 1
        SOURCE_RESTRICTION_DISABLED = 2
    identities = _messages.StringField(1, repeated=True)
    identityType = _messages.EnumField('IdentityTypeValueValuesEnum', 2)
    sourceRestriction = _messages.EnumField('SourceRestrictionValueValuesEnum', 3)
    sources = _messages.MessageField('GoogleIdentityAccesscontextmanagerV1EgressSource', 4, repeated=True)