from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventTypeValueValuesEnum(_messages.Enum):
    """Optional. The type of this transaction event.

    Values:
      TRANSACTION_EVENT_TYPE_UNSPECIFIED: Default, unspecified event type.
      MERCHANT_APPROVE: Indicates that the transaction is approved by the
        merchant. The accompanying reasons can include terms such as
        'INHOUSE', 'ACCERTIFY', 'CYBERSOURCE', or 'MANUAL_REVIEW'.
      MERCHANT_DENY: Indicates that the transaction is denied and concluded
        due to risks detected by the merchant. The accompanying reasons can
        include terms such as 'INHOUSE', 'ACCERTIFY', 'CYBERSOURCE', or
        'MANUAL_REVIEW'.
      MANUAL_REVIEW: Indicates that the transaction is being evaluated by a
        human, due to suspicion or risk.
      AUTHORIZATION: Indicates that the authorization attempt with the card
        issuer succeeded.
      AUTHORIZATION_DECLINE: Indicates that the authorization attempt with the
        card issuer failed. The accompanying reasons can include Visa's '54'
        indicating that the card is expired, or '82' indicating that the CVV
        is incorrect.
      PAYMENT_CAPTURE: Indicates that the transaction is completed because the
        funds were settled.
      PAYMENT_CAPTURE_DECLINE: Indicates that the transaction could not be
        completed because the funds were not settled.
      CANCEL: Indicates that the transaction has been canceled. Specify the
        reason for the cancellation. For example, 'INSUFFICIENT_INVENTORY'.
      CHARGEBACK_INQUIRY: Indicates that the merchant has received a
        chargeback inquiry due to fraud for the transaction, requesting
        additional information before a fraud chargeback is officially issued
        and a formal chargeback notification is sent.
      CHARGEBACK_ALERT: Indicates that the merchant has received a chargeback
        alert due to fraud for the transaction. The process of resolving the
        dispute without involving the payment network is started.
      FRAUD_NOTIFICATION: Indicates that a fraud notification is issued for
        the transaction, sent by the payment instrument's issuing bank because
        the transaction appears to be fraudulent. We recommend including TC40
        or SAFE data in the `reason` field for this event type. For partial
        chargebacks, we recommend that you include an amount in the `value`
        field.
      CHARGEBACK: Indicates that the merchant is informed by the payment
        network that the transaction has entered the chargeback process due to
        fraud. Reason code examples include Discover's '6005' and '6041'. For
        partial chargebacks, we recommend that you include an amount in the
        `value` field.
      CHARGEBACK_REPRESENTMENT: Indicates that the transaction has entered the
        chargeback process due to fraud, and that the merchant has chosen to
        enter representment. Reason examples include Discover's '6005' and
        '6041'. For partial chargebacks, we recommend that you include an
        amount in the `value` field.
      CHARGEBACK_REVERSE: Indicates that the transaction has had a fraud
        chargeback which was illegitimate and was reversed as a result. For
        partial chargebacks, we recommend that you include an amount in the
        `value` field.
      REFUND_REQUEST: Indicates that the merchant has received a refund for a
        completed transaction. For partial refunds, we recommend that you
        include an amount in the `value` field. Reason example: 'TAX_EXEMPT'
        (partial refund of exempt tax)
      REFUND_DECLINE: Indicates that the merchant has received a refund
        request for this transaction, but that they have declined it. For
        partial refunds, we recommend that you include an amount in the
        `value` field. Reason example: 'TAX_EXEMPT' (partial refund of exempt
        tax)
      REFUND: Indicates that the completed transaction was refunded by the
        merchant. For partial refunds, we recommend that you include an amount
        in the `value` field. Reason example: 'TAX_EXEMPT' (partial refund of
        exempt tax)
      REFUND_REVERSE: Indicates that the completed transaction was refunded by
        the merchant, and that this refund was reversed. For partial refunds,
        we recommend that you include an amount in the `value` field.
    """
    TRANSACTION_EVENT_TYPE_UNSPECIFIED = 0
    MERCHANT_APPROVE = 1
    MERCHANT_DENY = 2
    MANUAL_REVIEW = 3
    AUTHORIZATION = 4
    AUTHORIZATION_DECLINE = 5
    PAYMENT_CAPTURE = 6
    PAYMENT_CAPTURE_DECLINE = 7
    CANCEL = 8
    CHARGEBACK_INQUIRY = 9
    CHARGEBACK_ALERT = 10
    FRAUD_NOTIFICATION = 11
    CHARGEBACK = 12
    CHARGEBACK_REPRESENTMENT = 13
    CHARGEBACK_REVERSE = 14
    REFUND_REQUEST = 15
    REFUND_DECLINE = 16
    REFUND = 17
    REFUND_REVERSE = 18