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
def AddEnableTcpConnections(parser):
    """Adds a --enable-tcp-connections flag to the given parser."""
    help_text = '  If set, workstations allow plain TCP connections.'
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--enable-tcp-connections', action='store_true', help=help_text)
    help_text = "  If set, workstations don't allow plain TCP connections."
    group.add_argument('--disable-tcp-connections', action='store_true', help=help_text)