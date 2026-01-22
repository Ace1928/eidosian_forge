from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import os
import signal
import sys
import threading
import time
import enum
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import multiline
from googlecloudsdk.core.console.style import parser
import six
def _PrintExitOutput(self, aborted=False, warned=False, failed=False):
    """Handles the final output for the progress tracker."""
    output_message = self._SetupExitOutput()
    if aborted:
        msg = self._aborted_message or 'Aborted.'
        self._header_stage.status = StageCompletionStatus.FAILED
    elif failed:
        msg = self._failure_message or 'Failed.'
        self._header_stage.status = StageCompletionStatus.FAILED
    elif warned:
        msg = self._warning_message or 'Completed with warnings:'
        self._header_stage.status = StageCompletionStatus.FAILED
    else:
        msg = self._success_message or 'Done.'
        self._header_stage.status = StageCompletionStatus.SUCCESS
    if self._done_message_callback:
        msg += ' ' + self._done_message_callback()
    self._UpdateMessage(output_message, msg)
    self._Print(self._symbols.interrupted)