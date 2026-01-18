from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.cloudresourcemanager import projects_util
from googlecloudsdk.command_lib.iam import iam_util
Sets the IAM policy for an organization.

    Args:
      organization_id: organization id.
      policy_file: A JSON or YAML file containing the IAM policy.

    Returns:
      The output from the SetIamPolicy API call.
    