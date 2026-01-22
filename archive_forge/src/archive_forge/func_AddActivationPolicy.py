from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
def AddActivationPolicy(parser, hidden=False):
    base.ChoiceArgument('--activation-policy', required=False, choices=['always', 'never'], default=None, hidden=hidden, help_str='Activation policy for this instance. This specifies when the instance should be activated and is applicable only when the instance state is `RUNNABLE`. The default is `always`. More information on activation policies can be found here: https://cloud.google.com/sql/docs/mysql/start-stop-restart-instance#activation_policy').AddToParser(parser)