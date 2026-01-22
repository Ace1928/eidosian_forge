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
def AddWorkstationPortField(parser):
    """Adds a workstation-port flag to the given parser."""
    help_text = '  The port on the workstation to which traffic should be sent.'
    parser.add_argument('workstation_port', type=int, help=help_text)