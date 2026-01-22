from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags as build_flags
def AddSubstitutions(argument_group):
    """Adds a substituion flag to the given argument group.

  Args:
    argument_group: argparse argument group to which the substitution flag will
      be added.
  """
    argument_group.add_argument('--substitutions', metavar='KEY=VALUE', type=arg_parsers.ArgDict(), help='Parameters to be substituted in the build specification. For example:\n\n  $ {command} ... --substitutions _FAVORITE_COLOR=blue,_NUM_CANDIES=10\n\nThis will result in a build where every occurrence of ```${_FAVORITE_COLOR}```\nin certain fields is replaced by "blue", and similarly for ```${_NUM_CANDIES}```\nand "10".\n\nSubstitutions can be applied to user-defined variables (starting with an\nunderscore) and to the following built-in variables: REPO_NAME, BRANCH_NAME,\nTAG_NAME, REVISION_ID, COMMIT_SHA, SHORT_SHA.\n\nFor more details, see:\nhttps://cloud.google.com/build/docs/configuring-builds/substitute-variable-values\n')