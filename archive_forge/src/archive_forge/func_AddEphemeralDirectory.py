from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddEphemeralDirectory(parser):
    spec = {'mount-path': str, 'disk-type': str, 'source-snapshot': str, 'source-image': str, 'read-only': bool}
    help_text = "  Ephemeral directory which won't persist across workstation sessions."
    parser.add_argument('--ephemeral-directory', type=arg_parsers.ArgDict(spec=spec), action='append', metavar='PROPERTY=VALUE', help=help_text)