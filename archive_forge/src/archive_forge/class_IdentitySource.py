from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IdentitySource(_messages.Message):
    """The VMware identity source in a private cloud. The identity source
  defines the Active Directory domain that you can configure and use for
  authentication. Currently, the only supported identity source type is Active
  Directory over LDAP.

  Enums:
    ApplianceTypeValueValuesEnum: Required. The appliance type of identity
      source. Can be vCenter or NSX-T.
    ProtocolValueValuesEnum: Required. The LDAP server connection protocol.
    StateValueValuesEnum: Output only. The state of identity source.

  Fields:
    anyDomainController: Any domain controller.
    applianceType: Required. The appliance type of identity source. Can be
      vCenter or NSX-T.
    baseGroupsDn: Required. The base distinguished name for groups.
    baseUsersDn: Required. The base distinguished name for users.
    createTime: Output only. Creation time of this resource.
    domain: Required. The domain name of the identity source.
    domainAlias: Optional. The domain alias of the identity source.
    domainPassword: Required. Input only. Input only and required. Password of
      the user in the domain who has a minimum of read-only access to the base
      distinguished names of users and groups.
    domainUser: Required. ID of a user in the domain who has a minimum of
      read-only access to the base distinguished names of users and groups.
    etag: Optional. Checksum that may be sent on update and delete requests to
      ensure that the user-provided value is up to date before the server
      processes a request. The server computes checksums based on the value of
      other fields in the request.
    name: Output only. Identifier. The resource name of this identity source.
      Resource names are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/us-central1-a/privateClouds/my-
      cloud/identitySources/my-identity-source`
    protocol: Required. The LDAP server connection protocol.
    specificDomainControllers: Specific domain controllers.
    sslCertificates: Optional. Input only. The root CA certificate files in
      CER format for the LDAPS server.
    state: Output only. The state of identity source.
    uid: Output only. System-generated unique identifier for the resource.
    updateTime: Output only. Last update time of this resource.
    vmwareIdentitySource: Output only. The name of the identity source in
      VMware vCenter.
  """

    class ApplianceTypeValueValuesEnum(_messages.Enum):
        """Required. The appliance type of identity source. Can be vCenter or
    NSX-T.

    Values:
      APPLIANCE_TYPE_UNSPECIFIED: The default value. This value should never
        be used.
      VCENTER: A vCenter appliance.
    """
        APPLIANCE_TYPE_UNSPECIFIED = 0
        VCENTER = 1

    class ProtocolValueValuesEnum(_messages.Enum):
        """Required. The LDAP server connection protocol.

    Values:
      PROTOCOL_UNSPECIFIED: The default value. This value should never be
        used.
      LDAP: A LDAP protocol.
      LDAPS: A LDAPS protocol.
    """
        PROTOCOL_UNSPECIFIED = 0
        LDAP = 1
        LDAPS = 2

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of identity source.

    Values:
      STATE_UNSPECIFIED: The default value. This value should never be used.
      ACTIVE: The identity source is ready.
      CREATING: The identity source is being created.
      DELETING: The identity source is being deleted.
      UPDATING: The identity source is being updated.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        CREATING = 2
        DELETING = 3
        UPDATING = 4
    anyDomainController = _messages.MessageField('AnyDomainController', 1)
    applianceType = _messages.EnumField('ApplianceTypeValueValuesEnum', 2)
    baseGroupsDn = _messages.StringField(3)
    baseUsersDn = _messages.StringField(4)
    createTime = _messages.StringField(5)
    domain = _messages.StringField(6)
    domainAlias = _messages.StringField(7)
    domainPassword = _messages.StringField(8)
    domainUser = _messages.StringField(9)
    etag = _messages.StringField(10)
    name = _messages.StringField(11)
    protocol = _messages.EnumField('ProtocolValueValuesEnum', 12)
    specificDomainControllers = _messages.MessageField('SpecificDomainControllers', 13)
    sslCertificates = _messages.StringField(14, repeated=True)
    state = _messages.EnumField('StateValueValuesEnum', 15)
    uid = _messages.StringField(16)
    updateTime = _messages.StringField(17)
    vmwareIdentitySource = _messages.StringField(18)