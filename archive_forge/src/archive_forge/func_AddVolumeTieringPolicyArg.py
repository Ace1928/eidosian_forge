from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.netapp import util as netapp_api_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddVolumeTieringPolicyArg(parser, messages):
    """Adds the --tiering-policy arg to the arg parser."""
    tiering_policy_arg_spec = {'tier-action': messages.TieringPolicy.TierActionValueValuesEnum, 'cooling-threshold-days': int}
    tiering_policy_help = "      Tiering Policy contains auto tiering policy on a volume.\n\n      Tiering Policy will have the following format\n      --tiering-policy=tier-action=TIER_ACTION,\n      cooling-threshold-days=COOLING_THRESHOLD_DAYS\n\n      tier-action is an enum, supported values are ENABLED or PAUSED,\ncooling-threshold-days is an integer represents time in days to mark the\nvolume's data block as cold and make it eligible for tiering,\ncan be range from 7-183. Default is 31.\n  "
    parser.add_argument('--tiering-policy', type=arg_parsers.ArgDict(spec=tiering_policy_arg_spec), metavar='tier-action=ENABLED|PAUSED', help=tiering_policy_help, hidden=True)