from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import resources
def wait_then_yield_nothing(operation, verb, client=None):
    """Blocks execution until an operation completes and does not yield a result.

  Args:
    operation (messages.Operation): The operation to wait on.
    verb (str): The verb to use in messages, such as "delete order".
    client (apitools.base.py.base_api.BaseApiService): API client for loading
        the results and operations clients.

  Returns:
    poller.GetResult(operation).
  """
    return _wait_for_operation(operation, verb, result_type=None, client=client)