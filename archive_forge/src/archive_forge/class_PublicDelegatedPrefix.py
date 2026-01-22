from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PublicDelegatedPrefix(_messages.Message):
    """A PublicDelegatedPrefix resource represents an IP block within a
  PublicAdvertisedPrefix that is configured within a single cloud scope
  (global or region). IPs in the block can be allocated to resources within
  that scope. Public delegated prefixes may be further broken up into smaller
  IP blocks in the same scope as the parent block.

  Enums:
    ByoipApiVersionValueValuesEnum: [Output Only] The version of BYOIP API.
    StatusValueValuesEnum: [Output Only] The status of the public delegated
      prefix, which can be one of following values: - `INITIALIZING` The
      public delegated prefix is being initialized and addresses cannot be
      created yet. - `READY_TO_ANNOUNCE` The public delegated prefix is a live
      migration prefix and is active. - `ANNOUNCED` The public delegated
      prefix is active. - `DELETING` The public delegated prefix is being
      deprovsioned.

  Fields:
    byoipApiVersion: [Output Only] The version of BYOIP API.
    creationTimestamp: [Output Only] Creation timestamp in RFC3339 text
      format.
    description: An optional description of this resource. Provide this
      property when you create the resource.
    fingerprint: Fingerprint of this resource. A hash of the contents stored
      in this object. This field is used in optimistic locking. This field
      will be ignored when inserting a new PublicDelegatedPrefix. An up-to-
      date fingerprint must be provided in order to update the
      PublicDelegatedPrefix, otherwise the request will fail with error 412
      conditionNotMet. To see the latest fingerprint, make a get() request to
      retrieve a PublicDelegatedPrefix.
    id: [Output Only] The unique identifier for the resource type. The server
      generates this identifier.
    ipCidrRange: The IP address range, in CIDR format, represented by this
      public delegated prefix.
    isLiveMigration: If true, the prefix will be live migrated.
    kind: [Output Only] Type of the resource. Always
      compute#publicDelegatedPrefix for public delegated prefixes.
    name: Name of the resource. Provided by the client when the resource is
      created. The name must be 1-63 characters long, and comply with RFC1035.
      Specifically, the name must be 1-63 characters long and match the
      regular expression `[a-z]([-a-z0-9]*[a-z0-9])?` which means the first
      character must be a lowercase letter, and all following characters must
      be a dash, lowercase letter, or digit, except the last character, which
      cannot be a dash.
    parentPrefix: The URL of parent prefix. Either PublicAdvertisedPrefix or
      PublicDelegatedPrefix.
    publicDelegatedSubPrefixs: The list of sub public delegated prefixes that
      exist for this public delegated prefix.
    region: [Output Only] URL of the region where the public delegated prefix
      resides. This field applies only to the region resource. You must
      specify this field as part of the HTTP request URL. It is not settable
      as a field in the request body.
    selfLink: [Output Only] Server-defined URL for the resource.
    status: [Output Only] The status of the public delegated prefix, which can
      be one of following values: - `INITIALIZING` The public delegated prefix
      is being initialized and addresses cannot be created yet. -
      `READY_TO_ANNOUNCE` The public delegated prefix is a live migration
      prefix and is active. - `ANNOUNCED` The public delegated prefix is
      active. - `DELETING` The public delegated prefix is being deprovsioned.
  """

    class ByoipApiVersionValueValuesEnum(_messages.Enum):
        """[Output Only] The version of BYOIP API.

    Values:
      V1: This public delegated prefix usually takes 4 weeks to delete, and
        the BGP status cannot be changed. Announce and Withdraw APIs can not
        be used on this prefix.
      V2: This public delegated prefix takes minutes to delete. Announce and
        Withdraw APIs can be used on this prefix to change the BGP status.
    """
        V1 = 0
        V2 = 1

    class StatusValueValuesEnum(_messages.Enum):
        """[Output Only] The status of the public delegated prefix, which can be
    one of following values: - `INITIALIZING` The public delegated prefix is
    being initialized and addresses cannot be created yet. -
    `READY_TO_ANNOUNCE` The public delegated prefix is a live migration prefix
    and is active. - `ANNOUNCED` The public delegated prefix is active. -
    `DELETING` The public delegated prefix is being deprovsioned.

    Values:
      ANNOUNCED: The public delegated prefix is active.
      ANNOUNCED_TO_GOOGLE: The prefix is announced within Google network.
      ANNOUNCED_TO_INTERNET: The prefix is announced to Internet and within
        Google.
      DELETING: The public delegated prefix is being deprovsioned.
      INITIALIZING: The public delegated prefix is being initialized and
        addresses cannot be created yet.
      READY_TO_ANNOUNCE: The public delegated prefix is currently withdrawn
        but ready to be announced.
    """
        ANNOUNCED = 0
        ANNOUNCED_TO_GOOGLE = 1
        ANNOUNCED_TO_INTERNET = 2
        DELETING = 3
        INITIALIZING = 4
        READY_TO_ANNOUNCE = 5
    byoipApiVersion = _messages.EnumField('ByoipApiVersionValueValuesEnum', 1)
    creationTimestamp = _messages.StringField(2)
    description = _messages.StringField(3)
    fingerprint = _messages.BytesField(4)
    id = _messages.IntegerField(5, variant=_messages.Variant.UINT64)
    ipCidrRange = _messages.StringField(6)
    isLiveMigration = _messages.BooleanField(7)
    kind = _messages.StringField(8, default='compute#publicDelegatedPrefix')
    name = _messages.StringField(9)
    parentPrefix = _messages.StringField(10)
    publicDelegatedSubPrefixs = _messages.MessageField('PublicDelegatedPrefixPublicDelegatedSubPrefix', 11, repeated=True)
    region = _messages.StringField(12)
    selfLink = _messages.StringField(13)
    status = _messages.EnumField('StatusValueValuesEnum', 14)