from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudcommerceconsumerprocurementProjectsFreeTrialsCreateRequest(_messages.Message):
    """A CloudcommerceconsumerprocurementProjectsFreeTrialsCreateRequest
  object.

  Fields:
    googleCloudCommerceConsumerProcurementV1alpha1FreeTrial: A
      GoogleCloudCommerceConsumerProcurementV1alpha1FreeTrial resource to be
      passed as the request body.
    parent: Required. The parent resource to query for FreeTrials. Currently
      the only parent supported is "projects/{project-id}".
  """
    googleCloudCommerceConsumerProcurementV1alpha1FreeTrial = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1FreeTrial', 1)
    parent = _messages.StringField(2, required=True)