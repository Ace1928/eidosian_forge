from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddInitialRolloutPhaseIDFlag():
    """Adds --initial-rollout-phase-id flag."""
    help_text = textwrap.dedent('  The phase to start the initial rollout at when creating the release.\n  The phase ID must be a valid phase on the rollout. If not specified, then the\n  rollout will start at the first phase.\n\n  Examples:\n\n  Start rollout at `stable` phase:\n\n    $ {command} --initial-rollout-phase-id=stable\n\n  ')
    return base.Argument('--initial-rollout-phase-id', help=help_text, hidden=False, default=None, required=False)