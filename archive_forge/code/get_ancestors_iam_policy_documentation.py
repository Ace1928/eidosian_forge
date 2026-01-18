from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resource_manager import folders
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iam import policies_flags
from googlecloudsdk.command_lib.resource_manager import flags
Get IAM policies for a folder and its ancestors.

  Get IAM policies for a folder and its ancestors, given a folder ID.

  ## EXAMPLES

  To get IAM policies for folder ``3589215982'' and its ancestors, run:

    $ {command} 3589215982
  