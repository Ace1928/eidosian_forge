from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LDAPSSettings(_messages.Message):
    """LDAPSSettings represents the ldaps settings for domain resource. LDAP is
  the Lightweight Directory Access Protocol, defined in
  https://tools.ietf.org/html/rfc4511. The settings object configures LDAP
  over SSL/TLS, whether it is over port 636 or the StartTLS operation. If
  LDAPSSettings is being changed, it will be placed into the UPDATING state,
  which indicates that the resource is being reconciled. At this point, Get
  will reflect an intermediate state.

  Enums:
    StateValueValuesEnum: Output only. The current state of this LDAPS
      settings.

  Fields:
    certificate: Output only. The certificate used to configure LDAPS.
      Certificates can be chained with a maximum length of 15.
    certificatePassword: Input only. The password used to encrypt the uploaded
      PFX certificate.
    certificatePfx: Input only. The uploaded PKCS12-formatted certificate to
      configure LDAPS with. It will enable the domain controllers in this
      domain to accept LDAPS connections (either LDAP over SSL/TLS or the
      StartTLS operation). A valid certificate chain must form a valid x.509
      certificate chain (or be comprised of a single self-signed certificate.
      It must be encrypted with either: 1) PBES2 + PBKDF2 + AES256 encryption
      and SHA256 PRF; or 2) pbeWithSHA1And3-KeyTripleDES-CBC Private key must
      be included for the leaf / single self-signed certificate. Note: For a
      fqdn your-example-domain.com, the wildcard fqdn is *.your-example-
      domain.com. Specifically the leaf certificate must have: - Either a
      blank subject or a subject with CN matching the wildcard fqdn. - Exactly
      two SANs - the fqdn and wildcard fqdn. - Encipherment and digital key
      signature key usages. - Server authentication extended key usage
      (OID=1.3.6.1.5.5.7.3.1) - Private key must be in one of the following
      formats: RSA, ECDSA, ED25519. - Private key must have appropriate key
      length: 2048 for RSA, 256 for ECDSA - Signature algorithm of the leaf
      certificate cannot be MD2, MD5 or SHA1.
    name: The resource name of the LDAPS settings. Uses the form:
      `projects/{project}/locations/{location}/domains/{domain}`.
    state: Output only. The current state of this LDAPS settings.
    updateTime: Output only. Last update time.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of this LDAPS settings.

    Values:
      STATE_UNSPECIFIED: Not Set
      UPDATING: The LDAPS setting is being updated.
      ACTIVE: The LDAPS setting is ready.
      FAILED: The LDAPS setting is not applied correctly.
    """
        STATE_UNSPECIFIED = 0
        UPDATING = 1
        ACTIVE = 2
        FAILED = 3
    certificate = _messages.MessageField('Certificate', 1)
    certificatePassword = _messages.StringField(2)
    certificatePfx = _messages.BytesField(3)
    name = _messages.StringField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)
    updateTime = _messages.StringField(6)