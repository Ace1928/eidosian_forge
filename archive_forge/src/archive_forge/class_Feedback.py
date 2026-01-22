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
@base.ReleaseTracks(base.ReleaseTrack.GA)
class Feedback(base.Command):
    """Provide feedback to the Google Cloud CLI team.

  The Google Cloud CLI team offers support through a number of channels:

  * Google Cloud CLI Issue Tracker
  * Stack Overflow "#gcloud" tag
  * google-cloud-dev Google group

  This command lists the available channels and facilitates getting help through
  one of them by opening a web browser to the relevant page, possibly with
  information relevant to the current install and configuration pre-populated in
  form fields on that page.
  """
    detailed_help = {'EXAMPLES': "\n          To send feedback, including the log file for the most recent command,\n          run:\n\n            $ {command}\n\n          To send feedback with a previously generated log file named\n          'my-logfile', run:\n\n            $ {command} --log-file=my-logfile\n          "}
    category = base.SDK_TOOLS_CATEGORY

    @staticmethod
    def Args(parser):
        parser.add_argument('--log-file', help='Path to the log file from a prior gcloud run.')

    def Run(self, args):
        info = info_holder.InfoHolder(anonymizer=info_holder.Anonymizer())
        log_data = None
        if args.log_file:
            try:
                log_data = info_holder.LogData.FromFile(args.log_file)
            except files.Error as err:
                log.warning('Error reading the specified file [{0}]: {1}\n'.format(args.log_file, err))
        if args.quiet:
            _PrintQuiet(six.text_type(info), log_data)
        else:
            log.status.Print(FEEDBACK_MESSAGE)
            if not log_data:
                log_data = _SuggestIncludeRecentLogs()
            if log_data or console_io.PromptContinue(prompt_string='No invocation selected. Would you still like to file a bug (will open a new browser tab)'):
                feedback_util.OpenNewIssueInBrowser(info, log_data)