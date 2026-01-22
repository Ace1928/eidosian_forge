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
def AddLabelsField(parser):
    """Adds a --labels flag to the given parser."""
    help_text = '  Labels that are applied to the configuration and propagated to the underlying\n  Compute Engine resources.\n\n  Example:\n\n    $ {command} --labels=label1=value1,label2=value2'
    parser.add_argument('--labels', metavar='LABELS', type=arg_parsers.ArgDict(key_type=str, value_type=str), help=help_text)