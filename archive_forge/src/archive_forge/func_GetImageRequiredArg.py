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
def GetImageRequiredArg():
    """Gets IMAGE required positional argument."""
    help_txt = textwrap.dedent('  A container image.\n\n  A valid container image has the format of\n    LOCATION-docker.pkg.dev/PROJECT-ID/REPOSITORY-ID/IMAGE\n\n  A valid container image that can be referenced by tag or digest, has the format of\n    LOCATION-docker.pkg.dev/PROJECT-ID/REPOSITORY-ID/IMAGE:tag\n    LOCATION-docker.pkg.dev/PROJECT-ID/REPOSITORY-ID/IMAGE@sha256:digest\n')
    return base.Argument('IMAGE', help=help_txt)