from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import json
import sys
from apitools.base.protorpclite import messages as apitools_messages
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py.exceptions import HttpBadRequestError
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import command_loading
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.apis import yaml_command_schema
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.util import files
import six
def _GetModifiedIamPolicyRemoveIamBinding(self, args, add_condition=False):
    """Get the IAM policy and remove the specified binding to it.

    Args:
      args: an argparse namespace.
      add_condition: True if support condition.

    Returns:
      IAM policy.
    """
    from googlecloudsdk.command_lib.iam import iam_util
    if add_condition:
        condition = iam_util.ValidateAndExtractCondition(args)
        policy = self._GetIamPolicy(args)
        iam_util.RemoveBindingFromIamPolicyWithCondition(policy, args.member, args.role, condition, all_conditions=args.all)
    else:
        policy = self._GetIamPolicy(args)
        iam_util.RemoveBindingFromIamPolicy(policy, args.member, args.role)
    return policy