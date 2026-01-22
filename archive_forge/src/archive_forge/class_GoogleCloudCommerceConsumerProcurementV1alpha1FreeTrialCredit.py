from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1FreeTrialCredit(_messages.Message):
    """Credit represents the real time credit information.

  Fields:
    creationDate: Date credit was created.
    endTime: When the credit expires. If empty then there's no upper bound of
      credit's effective timespan (i.e. the credit never expires).
    remainingAmount: The amount of the credit remaining.
    startTime: When the credit becomes effective. If empty then there's no
      lower bound of credit's effective timespan (i.e. the credit becomes
      effective at the time of its creation). For credit creation, this cannot
      be in the past.
    value: The value of the credit.
  """
    creationDate = _messages.StringField(1)
    endTime = _messages.StringField(2)
    remainingAmount = _messages.MessageField('GoogleTypeMoney', 3)
    startTime = _messages.StringField(4)
    value = _messages.MessageField('GoogleTypeMoney', 5)