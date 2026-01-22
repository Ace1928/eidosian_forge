from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1CheckConsentResponse(_messages.Message):
    """Response for check consent.

  Fields:
    consent: The Consent for this agreement if a consent is active.
  """
    consent = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1Consent', 1)