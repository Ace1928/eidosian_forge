from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import textwrap
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib import feedback_util
from googlecloudsdk.command_lib import info_holder
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import text as text_util
import six
from six.moves import map
def _PrintQuiet(info_str, log_data):
    """Print message referring to various feedback resources for quiet execution.

  Args:
    info_str: str, the output of `gcloud info`
    log_data: info_holder.LogData, log data for the provided log file
  """
    if log_data:
        if not log_data.traceback:
            log.Print('Please consider including the log file [{0}] in any feedback you submit.'.format(log_data.filename))
    log.Print(textwrap.dedent('\n      If you have a question, post it on Stack Overflow using the "gcloud" tag\n      at [{0}].\n\n      For general feedback, use our groups page\n      [{1}],\n      send a mail to [google-cloud-dev@googlegroups.com], or visit the [#gcloud]\n      IRC channel on freenode.\n\n      If you have found a bug, file it using our issue tracker site at\n      [{2}].\n\n      Please include the following information when filing a bug report:      ').format(STACKOVERFLOW_URL, GROUPS_PAGE_URL, feedback_util.ISSUE_TRACKER_URL))
    divider = feedback_util.GetDivider()
    log.Print(divider)
    if log_data and log_data.traceback:
        log.Print(log_data.traceback)
    log.Print(info_str.strip())
    log.Print(divider)