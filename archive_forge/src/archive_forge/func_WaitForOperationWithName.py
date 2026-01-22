from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.firestore import api_utils
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import resources
def WaitForOperationWithName(operation_name):
    """Waits for the given Operation to complete."""
    operation = api_utils.GetMessages().GoogleLongrunningOperation(name=operation_name)
    return WaitForOperation(operation)