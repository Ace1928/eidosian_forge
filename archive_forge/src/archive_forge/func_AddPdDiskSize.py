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
def AddPdDiskSize(parser):
    """Adds a --pd-disk-size flag to the given parser."""
    help_text = '  Size of the persistent directory in GB.'
    parser.add_argument('--pd-disk-size', choices=[10, 50, 100, 200, 500, 1000], default=200, type=int, help=help_text)