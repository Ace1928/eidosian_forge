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
def AddEnableAuditAgent(parser, use_default=True):
    """Adds an --enable-audit-agent flag to the given parser."""
    help_text = '  Whether to enable Linux `auditd` logging on the workstation. When enabled,\n  a service account must also be specified that has `logging.buckets.write`\n  permission on the project.'
    parser.add_argument('--enable-audit-agent', action='store_true', default=False if use_default else None, help=help_text)