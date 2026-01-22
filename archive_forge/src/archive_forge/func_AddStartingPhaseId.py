from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddStartingPhaseId(parser):
    """Add --starting-phase-id flag."""
    help_text = textwrap.dedent('  If set, starts the created rollout at the specified phase.\n\n  Start rollout at `stable` phase:\n\n    $ {command} --starting-phase-id=stable\n\n  ')
    parser.add_argument('--starting-phase-id', help=help_text, hidden=False, default=None, required=False)