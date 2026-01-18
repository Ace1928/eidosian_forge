from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import exceptions
Removes an IAM Policy binding from a Subscription.

    Args:
      subscription_ref (Resource): Resource reference for subscription to remove
        IAM policy binding from.
      member (str): The member to add.
      role (str): The role to assign to the member.

    Returns:
      Policy: the updated policy.
    Raises:
      api_exception.HttpException: If either of the requests failed.
    