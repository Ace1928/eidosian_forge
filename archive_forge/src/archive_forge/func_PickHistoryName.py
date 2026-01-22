from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import arg_util
from googlecloudsdk.api_lib.firebase.test import ctrl_c_handler
from googlecloudsdk.api_lib.firebase.test import exit_code
from googlecloudsdk.api_lib.firebase.test import history_picker
from googlecloudsdk.api_lib.firebase.test import matrix_ops
from googlecloudsdk.api_lib.firebase.test import results_bucket
from googlecloudsdk.api_lib.firebase.test import results_summary
from googlecloudsdk.api_lib.firebase.test import tool_results
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.api_lib.firebase.test.ios import arg_manager
from googlecloudsdk.api_lib.firebase.test.ios import matrix_creator
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
import six
def PickHistoryName(args):
    """Returns the results history name to use to look up a history ID.

  The history ID corresponds to a history name. If the user provides their own
  history name, we use that to look up the history ID; Otherwise, we punt and
  let the Testing service determine the appropriate history ID to publish to.

  Args:
    args: an argparse namespace. All the arguments that were provided to the
      command invocation (i.e. group and command arguments combined).

  Returns:
    Either a string containing a history name derived from user-supplied data,
    or None if we lack the required information.
  """
    if args.results_history_name:
        return args.results_history_name
    return None