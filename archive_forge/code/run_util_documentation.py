from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.functions.v2 import util as api_util
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.command_lib.run import connection_context
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.core import resources
Add the IAM binding for the invoker role on the function's Cloud Run service.

  Args:
    function: cloudfunctions_v2_messages.Function, a GCF v2 function.
    member: str, The user to bind the Invoker role to.
    add_binding: bool, Whether to add to or remove from the IAM policy.

  Returns:
    A google.iam.v1.Policy
  