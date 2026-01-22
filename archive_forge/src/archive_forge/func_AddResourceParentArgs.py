from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from enum import Enum
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
def AddResourceParentArgs(parser, noun, verb):
    """Adds project, folder, and organization flags to the parser."""
    parent_resource_group = parser.add_group(help='      The scope of the {}. If a scope is not specified, the current project is\n      used as the default.'.format(noun), mutex=True)
    common_args.ProjectArgument(help_text_to_prepend='The project of the {} {}.'.format(noun, verb), help_text_to_overwrite="      The project name to use. If a project name is not specified, then the\n      current project is used. The current project can be listed using gcloud\n      config list --format='text(core.project)' and can be set using gcloud\n      config set project PROJECTID.\n\n      `--project` and its fallback `{core_project}` property play two roles. It\n      specifies the project of the resource to operate on, and also specifies\n      the project for API enablement check, quota, and billing. To specify a\n      different project for quota and billing, use `--billing-project` or\n      `{billing_project}` property.\n      ".format(core_project=properties.VALUES.core.project, billing_project=properties.VALUES.billing.quota_project)).AddToParser(parent_resource_group)
    parent_resource_group.add_argument('--folder', metavar='FOLDER_ID', type=str, help='The folder of the {} {}.'.format(noun, verb))
    parent_resource_group.add_argument('--organization', metavar='ORGANIZATION_ID', type=str, help='The organization of the {} {}.'.format(noun, verb))