from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddInitialRolloutAnnotationsFlag():
    """Adds --initial-rollout-annotations flag."""
    help_text = textwrap.dedent('  Annotations to apply to the initial rollout when creating the release.\n  Annotations take the form of key/value string pairs.\n\n  Examples:\n\n  Add annotations:\n\n    $ {command} --initial-rollout-annotations="from_target=test,status=stable"\n\n  ')
    return base.Argument('--initial-rollout-annotations', help=help_text, metavar='KEY=VALUE', type=arg_parsers.ArgDict())