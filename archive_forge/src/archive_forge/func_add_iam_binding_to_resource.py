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
def add_iam_binding_to_resource(args, url, messages, policy, task_type):
    """Extracts new binding from args and applies to existing policy.

  Args:
    args (argparse Args): Contains flags user ran command with.
    url (CloudUrl): URL of target resource, already validated for type.
    messages (object): Must contain IAM data types needed to create new policy.
    policy (object): Existing IAM policy on target to update.
    task_type (set_iam_policy_task._SetIamPolicyTask): The task instance to use
      to execute the iam binding change.

  Returns:
    object: The updated IAM policy set in the cloud.
  """
    condition = iam_util.ValidateAndExtractCondition(args)
    iam_util.AddBindingToIamPolicyWithCondition(messages.Policy.BindingsValueListEntry, messages.Expr, policy, args.member, args.role, condition)
    task_output = task_type(url, policy).execute()
    return task_util.get_first_matching_message_payload(task_output.messages, task.Topic.SET_IAM_POLICY)