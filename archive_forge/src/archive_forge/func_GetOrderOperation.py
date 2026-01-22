from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.core import properties
@staticmethod
def GetOrderOperation(operation_name):
    """Calls the Procurement Consumer Orders.Operations.Get method.

    Args:
      operation_name: Name of the order operation.

    Returns:
      Order operation.
    """
    request = GetMessagesModule().CloudcommerceconsumerprocurementBillingAccountsOrdersOperationsGetRequest(name=operation_name)
    try:
        return Operations.GetOrderOperationService().Get(request)
    except apitools_exceptions.HttpError as error:
        raise exceptions.HttpException(error)