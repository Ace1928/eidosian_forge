from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AccessReason(_messages.Message):
    """A AccessReason object.

  Enums:
    TypeValueValuesEnum: Type of access justification.

  Fields:
    detail: More detail about certain reason types. See comments for each type
      above.
    type: Type of access justification.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Type of access justification.

    Values:
      TYPE_UNSPECIFIED: Default value for proto, shouldn't be used.
      CUSTOMER_INITIATED_SUPPORT: Customer made a request or raised an issue
        that required the principal to access customer data. `detail` is of
        the form ("#####" is the issue ID): * "Feedback Report: #####" * "Case
        Number: #####" * "Case ID: #####" * "E-PIN Reference: #####" *
        "Google-#####" * "T-#####"
      GOOGLE_INITIATED_SERVICE: The principal accessed customer data in order
        to diagnose or resolve a suspected issue in services. Often this
        access is used to confirm that customers are not affected by a
        suspected service issue or to remediate a reversible system issue.
      GOOGLE_INITIATED_REVIEW: Google initiated service for security, fraud,
        abuse, or compliance purposes.
      THIRD_PARTY_DATA_REQUEST: The principal was compelled to access customer
        data in order to respond to a legal third party data request or
        process, including legal processes from customers themselves.
      GOOGLE_RESPONSE_TO_PRODUCTION_ALERT: The principal accessed customer
        data in order to diagnose or resolve a suspected issue in services or
        a known outage.
      CLOUD_INITIATED_ACCESS: Similar to 'GOOGLE_INITIATED_SERVICE' or
        'GOOGLE_INITIATED_REVIEW', but with universe agnostic naming. The
        principal accessed customer data in order to diagnose or resolve a
        suspected issue in services or a known outage, or for security, fraud,
        abuse, or compliance review purposes.
    """
        TYPE_UNSPECIFIED = 0
        CUSTOMER_INITIATED_SUPPORT = 1
        GOOGLE_INITIATED_SERVICE = 2
        GOOGLE_INITIATED_REVIEW = 3
        THIRD_PARTY_DATA_REQUEST = 4
        GOOGLE_RESPONSE_TO_PRODUCTION_ALERT = 5
        CLOUD_INITIATED_ACCESS = 6
    detail = _messages.StringField(1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)