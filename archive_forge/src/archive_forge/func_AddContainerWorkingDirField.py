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
def AddContainerWorkingDirField(parser):
    """Adds a --container-working-dir flag to the given parser."""
    help_text = '  If set, overrides the default DIR specified by the image.'
    parser.add_argument('--container-working-dir', help=help_text)