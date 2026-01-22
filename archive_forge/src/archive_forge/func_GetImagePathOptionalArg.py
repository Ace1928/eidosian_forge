from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
import textwrap
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetImagePathOptionalArg():
    """Gets IMAGE_PATH optional positional argument."""
    help_txt = textwrap.dedent('  An Artifact Registry repository or a container image.\n  If not specified, default config values are used.\n\n  A valid docker repository has the format of\n    LOCATION-docker.pkg.dev/PROJECT-ID/REPOSITORY-ID\n\n  A valid image has the format of\n    LOCATION-docker.pkg.dev/PROJECT-ID/REPOSITORY-ID/IMAGE_PATH\n')
    return base.Argument('IMAGE_PATH', help=help_txt, nargs='?')