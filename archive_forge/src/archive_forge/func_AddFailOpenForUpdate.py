from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import actions as calliope_actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddFailOpenForUpdate(parser):
    """Adds failOpen flag to the argparse.ArgumentParser."""
    parser.add_argument('--fail-open', action='store_true', default=None, help='      If enabled, the Interconnect will be configured with a should-secure\n      MACsec security policy, that allows the Google router to fallback to\n      cleartext traffic if the MKA session cannot be established. By default,\n      the Interconnect will be configured with a must-secure security policy\n      that drops all traffic if the MKA session cannot be established with your\n      router. Use --no-fail-open to disable it.\n      ')