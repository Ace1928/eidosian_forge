from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
import textwrap
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.command_lib.iam import iam_util as core_iam_util
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def AddUriFlags(parser, collection, api_version=None):
    """Adds `--uri` flag to the parser object for list commands.

  Args:
    parser: The argparse parser.
    collection: str, The resource collection name.
    api_version: str, The API version.
  """

    def _GetResourceUri(resource):
        updated = resources.REGISTRY.ParseRelativeName(resource.name, collection=collection, api_version=api_version)
        return updated.SelfLink()
    parser.display_info.AddUriFunc(_GetResourceUri)