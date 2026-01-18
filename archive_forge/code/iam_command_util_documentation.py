from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.command_lib.storage.tasks import task_executor
from googlecloudsdk.command_lib.storage.tasks import task_graph_executor
from googlecloudsdk.command_lib.storage.tasks import task_status
from googlecloudsdk.command_lib.storage.tasks import task_util
Extracts binding from args and removes it from existing policy.

  Args:
    args (argparse Args): Contains flags user ran command with.
    url (CloudUrl): URL of target resource, already validated for type.
    policy (object): Existing IAM policy on target to update.
    task_type (set_iam_policy_task._SetIamPolicyTask): The task instance to use
      to execute the iam binding change.

  Returns:
    object: The updated IAM policy set in the cloud.
  