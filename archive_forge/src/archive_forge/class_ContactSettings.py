from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContactSettings(_messages.Message):
    """Defines the contact information associated with a `Registration`.
  [ICANN](https://icann.org/) requires all domain names to have associated
  contact information. The `registrant_contact` is considered the domain's
  legal owner, and often the other contacts are identical.

  Enums:
    PrivacyValueValuesEnum: Required. Privacy setting for the contacts
      associated with the `Registration`.

  Fields:
    adminContact: Required. The administrative contact for the `Registration`.
    privacy: Required. Privacy setting for the contacts associated with the
      `Registration`.
    registrantContact: Required. The registrant contact for the
      `Registration`. *Caution: Anyone with access to this email address,
      phone number, and/or postal address can take control of the domain.*
      *Warning: For new `Registration`s, the registrant receives an email
      confirmation that they must complete within 15 days to avoid domain
      suspension.*
    technicalContact: Required. The technical contact for the `Registration`.
  """

    class PrivacyValueValuesEnum(_messages.Enum):
        """Required. Privacy setting for the contacts associated with the
    `Registration`.

    Values:
      CONTACT_PRIVACY_UNSPECIFIED: The contact privacy settings are undefined.
      PUBLIC_CONTACT_DATA: All the data from `ContactSettings` is publicly
        available. When setting this option, you must also provide a
        `PUBLIC_CONTACT_DATA_ACKNOWLEDGEMENT` in the `contact_notices` field
        of the request.
      PRIVATE_CONTACT_DATA: Deprecated: For more information, see [Cloud
        Domains feature deprecation](https://cloud.google.com/domains/docs/dep
        recations/feature-deprecations). None of the data from
        `ContactSettings` is publicly available. Instead, proxy contact data
        is published for your domain. Email sent to the proxy email address is
        forwarded to the registrant's email address. Cloud Domains provides
        this privacy proxy service at no additional cost.
      REDACTED_CONTACT_DATA: The organization name (if provided) and limited
        non-identifying data from `ContactSettings` is available to the public
        (e.g. country and state). The remaining data is marked as `REDACTED
        FOR PRIVACY` in the WHOIS database. The actual information redacted
        depends on the domain. For details, see [the registration privacy
        article](https://support.google.com/domains/answer/3251242).
    """
        CONTACT_PRIVACY_UNSPECIFIED = 0
        PUBLIC_CONTACT_DATA = 1
        PRIVATE_CONTACT_DATA = 2
        REDACTED_CONTACT_DATA = 3
    adminContact = _messages.MessageField('Contact', 1)
    privacy = _messages.EnumField('PrivacyValueValuesEnum', 2)
    registrantContact = _messages.MessageField('Contact', 3)
    technicalContact = _messages.MessageField('Contact', 4)