from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1Consent(_messages.Message):
    """A consent resource represents the relationship between a user and an
  agreement.

  Enums:
    StateValueValuesEnum: Output only. State of current consent.

  Fields:
    agreement: Full name of the agreement that was agreed to for this consent,
      ## in the format of one of: "commerceoffercatalog.googleapis.com/billing
      Accounts/{billing_account}/offers/{offer_id}/agreements/{agreement_id}".
      "commerceoffercatalog.googleapis.com/services/{service}/standardOffers/{
      offer_id}/agreements/{agreement_id}".
    agreementDocument: Full name of the agreement document that was agreed to
      for this consent, ## in the format of one of: commerceoffercatalog.googl
      eapis.com/billingAccounts/{billing_account}/offers/{offer_id}/agreements
      /{agreement_id}/documents/{document_id}
    createTime: Output only. The creation time of current consent.
    financialContract: Financial contracts linked to this consent.
    name: The resource name of a consent. An examples of valid names would be
      in the format of: -
      "billingAccounts/{billing_account}/consents/{consent}". -
      "projects/{project_number}/consents/{consent}".
    offer: The name of the offer linked to this consent. It is in the format
      of: "commerceoffercatalog.googleapis.com/billingAccounts/{billing_accoun
      t}/offers/{offer_id}". "commerceoffercatalog.googleapis.com/services/{se
      rvice}/standardOffers/{offer_id}".
    state: Output only. State of current consent.
    updateTime: Output only. The update time of current consent.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of current consent.

    Values:
      STATE_UNSPECIFIED: Unspecified value for the state. Sentinel value; do
        not use.
      ACTIVE: Represent the approved state of the consent.
      REVOKED: Represent the revoked state of the consent.
      ROLLEDBACK: Represent the rolled back state of the consent.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        REVOKED = 2
        ROLLEDBACK = 3
    agreement = _messages.StringField(1)
    agreementDocument = _messages.StringField(2)
    createTime = _messages.StringField(3)
    financialContract = _messages.StringField(4)
    name = _messages.StringField(5)
    offer = _messages.StringField(6)
    state = _messages.EnumField('StateValueValuesEnum', 7)
    updateTime = _messages.StringField(8)