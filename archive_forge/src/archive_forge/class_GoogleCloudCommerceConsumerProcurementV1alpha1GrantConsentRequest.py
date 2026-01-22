from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1GrantConsentRequest(_messages.Message):
    """Request message to grant consent.

  Fields:
    consent: Required. A consent to be granted. Set only agreement_document
      field.
    validateOnly: Optional. Flag is used to dry run grant consent behavior for
      the VMs and K8s products. If set, returns empty consent if consent can
      be granted. If unset, grants consent if consent can be granted.
  """
    consent = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1Consent', 1)
    validateOnly = _messages.BooleanField(2)