from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddAllowedPSCProjects(parser):
    """Adds the `--allowed-psc-projects` flag to the parser."""
    parser.add_argument('--allowed-psc-projects', required=False, type=arg_parsers.ArgList(), metavar='ALLOWED_PSC_PROJECTS', help='Comma-separated list of allowed consumer projects to create endpoints for Private Service Connect (PSC) connectivity for the instance. Only instances in PSC-enabled clusters are allowed to set this field.(e.g., `--allowed-psc-projects=project1,12345678,project2)')