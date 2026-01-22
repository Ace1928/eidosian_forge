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
def GetRawPredictRequestArg():
    """Adds arguments for raw-predict requests."""
    return base.Argument('--request', required=True, help="      The request to send to the endpoint.\n\n      If the request starts with the letter '*@*', the rest should be a file\n      name to read the request from, or '*@-*' to read from *stdin*. If the\n      request body actually starts with '*@*', it must be placed in a file.\n\n      If required, the *Content-Type* header should also be set appropriately,\n      particularly for binary data.\n      ")