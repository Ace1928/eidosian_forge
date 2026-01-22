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
def AddCommandField(parser):
    """Adds a --command flag to the given parser."""
    help_text = '      A command to run on the workstation.\n\n      Runs the command on the target workstation and then exits.\n      '
    parser.add_argument('--command', type=str, help=help_text)