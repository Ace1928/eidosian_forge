from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import signal
import sys
import time
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_completer
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import module_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
def _RunProgressTracker(self, args):
    start_time = time.time()

    def message_callback():
        remaining_time = args.progress_tracker - (time.time() - start_time)
        return '{0:.1f}s remaining'.format(remaining_time)
    with progress_tracker.ProgressTracker(message='This is a progress tracker.', detail_message_callback=message_callback):
        time.sleep(args.progress_tracker)