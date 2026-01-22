from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.core import properties
def GetQuoteChangeTypeEnum(raw_input):
    """Converts raw input to quote change type.

  Args:
    raw_input: Raw input of the quote change type.

  Returns:
    Converted quote change type.
  Raises:
    ValueError: The raw input is not recognized as a valid change type.
  """
    if raw_input == 'UPDATE':
        return GetMessagesModule().GoogleCloudCommerceConsumerProcurementV1alpha1ModifyQuoteOrderRequest().ChangeTypeValueValuesEnum.QUOTE_CHANGE_TYPE_UPDATE
    elif raw_input == 'CANCEL':
        return GetMessagesModule().GoogleCloudCommerceConsumerProcurementV1alpha1ModifyQuoteOrderRequest().ChangeTypeValueValuesEnum.QUOTE_CHANGE_TYPE_CANCEL
    elif raw_input == 'REVERT_CANCELLATION':
        return GetMessagesModule().GoogleCloudCommerceConsumerProcurementV1alpha1ModifyQuoteOrderRequest().ChangeTypeValueValuesEnum.QUOTE_CHANGE_TYPE_REVERT_CANCELLATION
    else:
        raise ValueError('Unrecognized quote change type {}.'.format(raw_input))