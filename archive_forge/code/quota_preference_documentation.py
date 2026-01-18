from apitools.base.py import list_pager
from googlecloudsdk.api_lib.quotas import message_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import common_args
Lists QuotaPreferences in a given project, folder or organization.

  Args:
    args: argparse.Namespace, The arguments that this command was invoked with.

  Returns:
    List of QuotaPreferences.
  