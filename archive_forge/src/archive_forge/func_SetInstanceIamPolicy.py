from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.spanner import databases
from googlecloudsdk.api_lib.spanner import instances
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
def SetInstanceIamPolicy(instance_ref, policy):
    """Sets the IAM policy on an instance."""
    msgs = apis.GetMessagesModule('spanner', 'v1')
    policy, field_mask = iam_util.ParsePolicyFileWithUpdateMask(policy, msgs.Policy)
    return instances.SetPolicy(instance_ref, policy, field_mask)