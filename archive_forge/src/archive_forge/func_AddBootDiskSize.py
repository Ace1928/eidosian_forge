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
def AddBootDiskSize(parser, use_default=True):
    """Adds a --boot-disk-size flag to the given parser."""
    help_text = '  Size of the boot disk in GB.'
    parser.add_argument('--boot-disk-size', default=50 if use_default else None, type=int, help=help_text)