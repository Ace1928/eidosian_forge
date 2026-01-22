from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Saml(_messages.Message):
    """Represents an SAML 2.0 identity provider.

  Fields:
    idpMetadataXml: Required. SAML identity provider (IdP) configuration
      metadata XML doc. The XML document must comply with the [SAML 2.0
      specification](https://docs.oasis-open.org/security/saml/v2.0/saml-
      metadata-2.0-os.pdf). The maximum size of an acceptable XML document is
      128K characters. The SAML metadata XML document must satisfy the
      following constraints: * Must contain an IdP Entity ID. * Must contain
      at least one non-expired signing certificate. * For each signing
      certificate, the expiration must be: * From no more than 7 days in the
      future. * To no more than 15 years in the future. * Up to three IdP
      signing keys are allowed. When updating the provider's metadata XML, at
      least one non-expired signing key must overlap with the existing
      metadata. This requirement is skipped if there are no non-expired
      signing keys present in the existing metadata.
  """
    idpMetadataXml = _messages.StringField(1)