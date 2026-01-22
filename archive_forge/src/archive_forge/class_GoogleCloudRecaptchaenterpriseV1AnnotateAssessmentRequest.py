from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1AnnotateAssessmentRequest(_messages.Message):
    """The request message to annotate an Assessment.

  Enums:
    AnnotationValueValuesEnum: Optional. The annotation that will be assigned
      to the Event. This field can be left empty to provide reasons that apply
      to an event without concluding whether the event is legitimate or
      fraudulent.
    ReasonsValueListEntryValuesEnum:

  Fields:
    accountId: Optional. A stable account identifier to apply to the
      assessment. This is an alternative to setting `account_id` in
      `CreateAssessment`, for example when a stable account identifier is not
      yet known in the initial request.
    annotation: Optional. The annotation that will be assigned to the Event.
      This field can be left empty to provide reasons that apply to an event
      without concluding whether the event is legitimate or fraudulent.
    hashedAccountId: Optional. A stable hashed account identifier to apply to
      the assessment. This is an alternative to setting `hashed_account_id` in
      `CreateAssessment`, for example when a stable account identifier is not
      yet known in the initial request.
    reasons: Optional. Reasons for the annotation that are assigned to the
      event.
    transactionEvent: Optional. If the assessment is part of a payment
      transaction, provide details on payment lifecycle events that occur in
      the transaction.
  """

    class AnnotationValueValuesEnum(_messages.Enum):
        """Optional. The annotation that will be assigned to the Event. This
    field can be left empty to provide reasons that apply to an event without
    concluding whether the event is legitimate or fraudulent.

    Values:
      ANNOTATION_UNSPECIFIED: Default unspecified type.
      LEGITIMATE: Provides information that the event turned out to be
        legitimate.
      FRAUDULENT: Provides information that the event turned out to be
        fraudulent.
      PASSWORD_CORRECT: Provides information that the event was related to a
        login event in which the user typed the correct password. Deprecated,
        prefer indicating CORRECT_PASSWORD through the reasons field instead.
      PASSWORD_INCORRECT: Provides information that the event was related to a
        login event in which the user typed the incorrect password.
        Deprecated, prefer indicating INCORRECT_PASSWORD through the reasons
        field instead.
    """
        ANNOTATION_UNSPECIFIED = 0
        LEGITIMATE = 1
        FRAUDULENT = 2
        PASSWORD_CORRECT = 3
        PASSWORD_INCORRECT = 4

    class ReasonsValueListEntryValuesEnum(_messages.Enum):
        """ReasonsValueListEntryValuesEnum enum type.

    Values:
      REASON_UNSPECIFIED: Default unspecified reason.
      CHARGEBACK: Indicates that the transaction had a chargeback issued with
        no other details. When possible, specify the type by using
        CHARGEBACK_FRAUD or CHARGEBACK_DISPUTE instead.
      CHARGEBACK_FRAUD: Indicates that the transaction had a chargeback issued
        related to an alleged unauthorized transaction from the cardholder's
        perspective (for example, the card number was stolen).
      CHARGEBACK_DISPUTE: Indicates that the transaction had a chargeback
        issued related to the cardholder having provided their card details
        but allegedly not being satisfied with the purchase (for example,
        misrepresentation, attempted cancellation).
      REFUND: Indicates that the completed payment transaction was refunded by
        the seller.
      REFUND_FRAUD: Indicates that the completed payment transaction was
        determined to be fraudulent by the seller, and was cancelled and
        refunded as a result.
      TRANSACTION_ACCEPTED: Indicates that the payment transaction was
        accepted, and the user was charged.
      TRANSACTION_DECLINED: Indicates that the payment transaction was
        declined, for example due to invalid card details.
      PAYMENT_HEURISTICS: Indicates the transaction associated with the
        assessment is suspected of being fraudulent based on the payment
        method, billing details, shipping address or other transaction
        information.
      INITIATED_TWO_FACTOR: Indicates that the user was served a 2FA
        challenge. An old assessment with `ENUM_VALUES.INITIATED_TWO_FACTOR`
        reason that has not been overwritten with `PASSED_TWO_FACTOR` is
        treated as an abandoned 2FA flow. This is equivalent to
        `FAILED_TWO_FACTOR`.
      PASSED_TWO_FACTOR: Indicates that the user passed a 2FA challenge.
      FAILED_TWO_FACTOR: Indicates that the user failed a 2FA challenge.
      CORRECT_PASSWORD: Indicates the user provided the correct password.
      INCORRECT_PASSWORD: Indicates the user provided an incorrect password.
      SOCIAL_SPAM: Indicates that the user sent unwanted and abusive messages
        to other users of the platform, such as spam, scams, phishing, or
        social engineering.
    """
        REASON_UNSPECIFIED = 0
        CHARGEBACK = 1
        CHARGEBACK_FRAUD = 2
        CHARGEBACK_DISPUTE = 3
        REFUND = 4
        REFUND_FRAUD = 5
        TRANSACTION_ACCEPTED = 6
        TRANSACTION_DECLINED = 7
        PAYMENT_HEURISTICS = 8
        INITIATED_TWO_FACTOR = 9
        PASSED_TWO_FACTOR = 10
        FAILED_TWO_FACTOR = 11
        CORRECT_PASSWORD = 12
        INCORRECT_PASSWORD = 13
        SOCIAL_SPAM = 14
    accountId = _messages.StringField(1)
    annotation = _messages.EnumField('AnnotationValueValuesEnum', 2)
    hashedAccountId = _messages.BytesField(3)
    reasons = _messages.EnumField('ReasonsValueListEntryValuesEnum', 4, repeated=True)
    transactionEvent = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1TransactionEvent', 5)