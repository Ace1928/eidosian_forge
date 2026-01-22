from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PublicAdvertisedPrefix(_messages.Message):
    """A public advertised prefix represents an aggregated IP prefix or
  netblock which customers bring to cloud. The IP prefix is a single unit of
  route advertisement and is announced globally to the internet.

  Enums:
    ByoipApiVersionValueValuesEnum: [Output Only] The version of BYOIP API.
    PdpScopeValueValuesEnum: Specifies how child public delegated prefix will
      be scoped. It could be one of following values: - `REGIONAL`: The public
      delegated prefix is regional only. The provisioning will take a few
      minutes. - `GLOBAL`: The public delegated prefix is global only. The
      provisioning will take ~4 weeks. - `GLOBAL_AND_REGIONAL` [output only]:
      The public delegated prefixes is BYOIP V1 legacy prefix. This is output
      only value and no longer supported in BYOIP V2.
    StatusValueValuesEnum: The status of the public advertised prefix.
      Possible values include: - `INITIAL`: RPKI validation is complete. -
      `PTR_CONFIGURED`: User has configured the PTR. - `VALIDATED`: Reverse
      DNS lookup is successful. - `REVERSE_DNS_LOOKUP_FAILED`: Reverse DNS
      lookup failed. - `PREFIX_CONFIGURATION_IN_PROGRESS`: The prefix is being
      configured. - `PREFIX_CONFIGURATION_COMPLETE`: The prefix is fully
      configured. - `PREFIX_REMOVAL_IN_PROGRESS`: The prefix is being removed.

  Fields:
    byoipApiVersion: [Output Only] The version of BYOIP API.
    creationTimestamp: [Output Only] Creation timestamp in RFC3339 text
      format.
    description: An optional description of this resource. Provide this
      property when you create the resource.
    dnsVerificationIp: The address to be used for reverse DNS verification.
    fingerprint: Fingerprint of this resource. A hash of the contents stored
      in this object. This field is used in optimistic locking. This field
      will be ignored when inserting a new PublicAdvertisedPrefix. An up-to-
      date fingerprint must be provided in order to update the
      PublicAdvertisedPrefix, otherwise the request will fail with error 412
      conditionNotMet. To see the latest fingerprint, make a get() request to
      retrieve a PublicAdvertisedPrefix.
    id: [Output Only] The unique identifier for the resource type. The server
      generates this identifier.
    ipCidrRange: The address range, in CIDR format, represented by this public
      advertised prefix.
    kind: [Output Only] Type of the resource. Always
      compute#publicAdvertisedPrefix for public advertised prefixes.
    name: Name of the resource. Provided by the client when the resource is
      created. The name must be 1-63 characters long, and comply with RFC1035.
      Specifically, the name must be 1-63 characters long and match the
      regular expression `[a-z]([-a-z0-9]*[a-z0-9])?` which means the first
      character must be a lowercase letter, and all following characters must
      be a dash, lowercase letter, or digit, except the last character, which
      cannot be a dash.
    pdpScope: Specifies how child public delegated prefix will be scoped. It
      could be one of following values: - `REGIONAL`: The public delegated
      prefix is regional only. The provisioning will take a few minutes. -
      `GLOBAL`: The public delegated prefix is global only. The provisioning
      will take ~4 weeks. - `GLOBAL_AND_REGIONAL` [output only]: The public
      delegated prefixes is BYOIP V1 legacy prefix. This is output only value
      and no longer supported in BYOIP V2.
    publicDelegatedPrefixs: [Output Only] The list of public delegated
      prefixes that exist for this public advertised prefix.
    selfLink: [Output Only] Server-defined URL for the resource.
    sharedSecret: [Output Only] The shared secret to be used for reverse DNS
      verification.
    status: The status of the public advertised prefix. Possible values
      include: - `INITIAL`: RPKI validation is complete. - `PTR_CONFIGURED`:
      User has configured the PTR. - `VALIDATED`: Reverse DNS lookup is
      successful. - `REVERSE_DNS_LOOKUP_FAILED`: Reverse DNS lookup failed. -
      `PREFIX_CONFIGURATION_IN_PROGRESS`: The prefix is being configured. -
      `PREFIX_CONFIGURATION_COMPLETE`: The prefix is fully configured. -
      `PREFIX_REMOVAL_IN_PROGRESS`: The prefix is being removed.
  """

    class ByoipApiVersionValueValuesEnum(_messages.Enum):
        """[Output Only] The version of BYOIP API.

    Values:
      V1: This public advertised prefix can be used to create both regional
        and global public delegated prefixes. It usually takes 4 weeks to
        create or delete a public delegated prefix. The BGP status cannot be
        changed.
      V2: This public advertised prefix can only be used to create regional
        public delegated prefixes. Public delegated prefix creation and
        deletion takes minutes and the BGP status can be modified.
    """
        V1 = 0
        V2 = 1

    class PdpScopeValueValuesEnum(_messages.Enum):
        """Specifies how child public delegated prefix will be scoped. It could
    be one of following values: - `REGIONAL`: The public delegated prefix is
    regional only. The provisioning will take a few minutes. - `GLOBAL`: The
    public delegated prefix is global only. The provisioning will take ~4
    weeks. - `GLOBAL_AND_REGIONAL` [output only]: The public delegated
    prefixes is BYOIP V1 legacy prefix. This is output only value and no
    longer supported in BYOIP V2.

    Values:
      GLOBAL: The public delegated prefix is global only. The provisioning
        will take ~4 weeks.
      GLOBAL_AND_REGIONAL: The public delegated prefixes is BYOIP V1 legacy
        prefix. This is output only value and no longer supported in BYOIP V2.
      REGIONAL: The public delegated prefix is regional only. The provisioning
        will take a few minutes.
    """
        GLOBAL = 0
        GLOBAL_AND_REGIONAL = 1
        REGIONAL = 2

    class StatusValueValuesEnum(_messages.Enum):
        """The status of the public advertised prefix. Possible values include: -
    `INITIAL`: RPKI validation is complete. - `PTR_CONFIGURED`: User has
    configured the PTR. - `VALIDATED`: Reverse DNS lookup is successful. -
    `REVERSE_DNS_LOOKUP_FAILED`: Reverse DNS lookup failed. -
    `PREFIX_CONFIGURATION_IN_PROGRESS`: The prefix is being configured. -
    `PREFIX_CONFIGURATION_COMPLETE`: The prefix is fully configured. -
    `PREFIX_REMOVAL_IN_PROGRESS`: The prefix is being removed.

    Values:
      ANNOUNCED_TO_INTERNET: The prefix is announced to Internet.
      INITIAL: RPKI validation is complete.
      PREFIX_CONFIGURATION_COMPLETE: The prefix is fully configured.
      PREFIX_CONFIGURATION_IN_PROGRESS: The prefix is being configured.
      PREFIX_REMOVAL_IN_PROGRESS: The prefix is being removed.
      PTR_CONFIGURED: User has configured the PTR.
      READY_TO_ANNOUNCE: The prefix is currently withdrawn but ready to be
        announced.
      REVERSE_DNS_LOOKUP_FAILED: Reverse DNS lookup failed.
      VALIDATED: Reverse DNS lookup is successful.
    """
        ANNOUNCED_TO_INTERNET = 0
        INITIAL = 1
        PREFIX_CONFIGURATION_COMPLETE = 2
        PREFIX_CONFIGURATION_IN_PROGRESS = 3
        PREFIX_REMOVAL_IN_PROGRESS = 4
        PTR_CONFIGURED = 5
        READY_TO_ANNOUNCE = 6
        REVERSE_DNS_LOOKUP_FAILED = 7
        VALIDATED = 8
    byoipApiVersion = _messages.EnumField('ByoipApiVersionValueValuesEnum', 1)
    creationTimestamp = _messages.StringField(2)
    description = _messages.StringField(3)
    dnsVerificationIp = _messages.StringField(4)
    fingerprint = _messages.BytesField(5)
    id = _messages.IntegerField(6, variant=_messages.Variant.UINT64)
    ipCidrRange = _messages.StringField(7)
    kind = _messages.StringField(8, default='compute#publicAdvertisedPrefix')
    name = _messages.StringField(9)
    pdpScope = _messages.EnumField('PdpScopeValueValuesEnum', 10)
    publicDelegatedPrefixs = _messages.MessageField('PublicAdvertisedPrefixPublicDelegatedPrefix', 11, repeated=True)
    selfLink = _messages.StringField(12)
    sharedSecret = _messages.StringField(13)
    status = _messages.EnumField('StatusValueValuesEnum', 14)