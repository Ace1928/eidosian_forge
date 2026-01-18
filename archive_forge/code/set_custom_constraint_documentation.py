from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as api_exceptions
from argcomplete import completers
from googlecloudsdk.api_lib.orgpolicy import service as org_policy_service
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.org_policies import exceptions
from googlecloudsdk.command_lib.org_policies import utils
from googlecloudsdk.core import log
Creates or updates a custom constraint from a JSON or YAML file.

    This first converts the contents of the specified file into a custom
    constraint object. It then fetches the current custom constraint using
    GetCustomConstraint. If it does not exist, the custom constraint is created
    using CreateCustomConstraint. If it does, the retrieved custom constraint is
    checked to see if it needs to be updated. If so, the custom constraint is
    updated using UpdateCustomConstraint.

    Args:
      args: argparse.Namespace, An object that contains the values for the
        arguments specified in the Args method.

    Returns:
      The created or updated custom constraint.
    