from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1FreeTrial(_messages.Message):
    """FreeTrial represents the free trial created for a specific offer and
  billing account with argentum. Free Trial resources are created by placing
  orders for 3p non-VM offers, or just enabling free trials for 1p offers and
  3p VM offers. Next Id: 7

  Fields:
    credit: Output only. Credit tracking the real time credit status.
    name: Output only. The resource name of the free trial item. This field is
      of the form: `projects/{project}/freeTrials/{free_trial}`. Present if
      free trial is created under the project's associated billing account for
      3p, or free trial is enabled for 1p product.
    productExternalName: External name for the product for which free trial
      exist. TODO(b/259732458) Mark this field "output only" once the standard
      offer migration completes.
    provider: Provider of the products for which free trial exist. Provider
      has the format of `providers/{provider_id}`.
    service: The one platform service name associated with the free trial.
      Format: 'services/{service_name}'.
  """
    credit = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1FreeTrialCredit', 1)
    name = _messages.StringField(2)
    productExternalName = _messages.StringField(3)
    provider = _messages.StringField(4)
    service = _messages.StringField(5)