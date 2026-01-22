from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegisterDomainRequest(_messages.Message):
    """Request for the `RegisterDomain` method.

  Enums:
    ContactNoticesValueListEntryValuesEnum:
    DomainNoticesValueListEntryValuesEnum:

  Fields:
    contactNotices: The list of contact notices that the caller acknowledges.
      The notices needed here depend on the values specified in
      `registration.contact_settings`.
    domainNotices: The list of domain notices that you acknowledge. Call
      `RetrieveRegisterParameters` to see the notices that need
      acknowledgement.
    registration: Required. The complete `Registration` resource to be
      created.
    validateOnly: When true, only validation is performed, without actually
      registering the domain. Follows:
      https://cloud.google.com/apis/design/design_patterns#request_validation
    yearlyPrice: Required. Yearly price to register or renew the domain. The
      value that should be put here can be obtained from
      RetrieveRegisterParameters or SearchDomains calls.
  """

    class ContactNoticesValueListEntryValuesEnum(_messages.Enum):
        """ContactNoticesValueListEntryValuesEnum enum type.

    Values:
      CONTACT_NOTICE_UNSPECIFIED: The notice is undefined.
      PUBLIC_CONTACT_DATA_ACKNOWLEDGEMENT: Required when setting the `privacy`
        field of `ContactSettings` to `PUBLIC_CONTACT_DATA`, which exposes
        contact data publicly.
    """
        CONTACT_NOTICE_UNSPECIFIED = 0
        PUBLIC_CONTACT_DATA_ACKNOWLEDGEMENT = 1

    class DomainNoticesValueListEntryValuesEnum(_messages.Enum):
        """DomainNoticesValueListEntryValuesEnum enum type.

    Values:
      DOMAIN_NOTICE_UNSPECIFIED: The notice is undefined.
      HSTS_PRELOADED: Indicates that the domain is preloaded on the HTTP
        Strict Transport Security list in browsers. Serving a website on such
        domain requires an SSL certificate. For details, see [how to get an
        SSL certificate](https://support.google.com/domains/answer/7638036).
    """
        DOMAIN_NOTICE_UNSPECIFIED = 0
        HSTS_PRELOADED = 1
    contactNotices = _messages.EnumField('ContactNoticesValueListEntryValuesEnum', 1, repeated=True)
    domainNotices = _messages.EnumField('DomainNoticesValueListEntryValuesEnum', 2, repeated=True)
    registration = _messages.MessageField('Registration', 3)
    validateOnly = _messages.BooleanField(4)
    yearlyPrice = _messages.MessageField('Money', 5)