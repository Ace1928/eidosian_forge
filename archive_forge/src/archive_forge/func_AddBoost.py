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
def AddBoost(parser):
    """Adds a --boost flag to the given parser."""
    help_text = '  Id of a boost configuration to start a workstations with.\n\n  Example:\n\n    $ {command} --boost=boost1'
    parser.add_argument('--boost', metavar='BOOST', type=str, help=help_text)