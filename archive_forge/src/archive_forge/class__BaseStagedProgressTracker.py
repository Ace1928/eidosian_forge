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
class _BaseStagedProgressTracker(collections_abc.Mapping):
    """Base class for staged progress trackers.

  During each tick, the tracker checks if there is a stage being displayed by
  checking if _stage_being_displayed is not None. If it is not none and stage
  has not completed, then the tracker will print an update. If the stage is
  done, then the tracker will write out the status of all completed stages
  in _running_stages_queue.
  """

    def __init__(self, message, stages, success_message, warning_message, failure_message, autotick, tick_delay, interruptable, aborted_message, tracker_id, done_message_callback, console=None):
        self._stages = collections.OrderedDict()
        for stage in stages:
            if stage.key in self._stages:
                raise ValueError('Duplicate stage key: {}'.format(stage.key))
            self._stages[stage.key] = stage
        self._stream = sys.stderr
        self._message = message
        self._success_message = success_message
        self._warning_message = warning_message
        self._failure_message = failure_message
        self._aborted_message = aborted_message
        self._done_message_callback = done_message_callback
        self._tracker_id = tracker_id
        if console is None:
            console = console_attr.GetConsoleAttr()
        console_width = console.GetTermSize()[0]
        if not isinstance(console_width, int) or console_width < 0:
            console_width = 0
        self._output_enabled = log.IsUserOutputEnabled() and console_width != 0
        self.__autotick = autotick and self._output_enabled
        self._interruptable = interruptable
        self._tick_delay = tick_delay
        self._symbols = console.GetProgressTrackerSymbols()
        self._done = False
        self._exception_is_uncaught = True
        self._ticks = 0
        self._ticker = None
        self._running_stages = set()
        self._completed_stages = []
        self._completed_with_warnings_stages = []
        self._exit_output_warnings = []
        self._lock = threading.Lock()

    def __getitem__(self, key):
        return self._stages[key]

    def __iter__(self):
        return iter(self._stages)

    def __len__(self):
        return len(self._stages)

    @property
    def _autotick(self):
        return self.__autotick

    def IsComplete(self, stage):
        """Returns True if the stage is complete."""
        return not (self.IsRunning(stage) or self.IsWaiting(stage))

    def IsRunning(self, stage):
        """Returns True if the stage is running."""
        stage = self._ValidateStage(stage, allow_complete=True)
        return stage.status == StageCompletionStatus.RUNNING

    def HasWarning(self):
        """Returns True if this tracker has encountered at least one warning."""
        return bool(self._exit_output_warnings)

    def IsWaiting(self, stage):
        """Returns True if the stage is not yet started."""
        stage = self._ValidateStage(stage, allow_complete=True)
        return stage.status == StageCompletionStatus.NOT_STARTED

    def _SetUpSignalHandler(self):
        """Sets up a signal handler for handling SIGINT."""

        def _CtrlCHandler(unused_signal, unused_frame):
            if self._interruptable:
                raise console_io.OperationCancelledError(self._aborted_message)
            else:
                self._NotifyUninterruptableError()
        try:
            self._old_signal_handler = signal.signal(signal.SIGINT, _CtrlCHandler)
            self._restore_old_handler = True
        except ValueError:
            self._restore_old_handler = False

    def _NotifyUninterruptableError(self):
        with self._lock:
            sys.stderr.write('\n\nThis operation cannot be cancelled.\n\n')

    def _TearDownSignalHandler(self):
        if self._restore_old_handler:
            try:
                signal.signal(signal.SIGINT, self._old_signal_handler)
            except ValueError:
                pass

    def __enter__(self):
        self._SetupOutput()
        self._SetUpSignalHandler()
        log.file_only_logger.info(self._message)
        self._Print()
        if self._autotick:

            def Ticker():
                while True:
                    _SleepSecs(self._tick_delay)
                    if self.Tick():
                        return
            self._ticker = threading.Thread(target=Ticker)
            self._ticker.daemon = True
            self._ticker.start()
        return self

    def __exit__(self, unused_ex_type, exc_value, unused_traceback):
        with self._lock:
            self._done = True
            if exc_value:
                if self._exception_is_uncaught:
                    self._HandleUncaughtException(exc_value)
            else:
                self._PrintExitOutput(warned=self.HasWarning())
        if self._ticker:
            self._ticker.join()
        self._TearDownSignalHandler()
        for warning_message in self._exit_output_warnings:
            log.status.Print('  %s' % warning_message)

    def _HandleUncaughtException(self, exc_value):
        if isinstance(exc_value, console_io.OperationCancelledError):
            self._PrintExitOutput(aborted=True)
        else:
            self._PrintExitOutput(failed=True)

    @abc.abstractmethod
    def _SetupOutput(self):
        """Sets up the output for the tracker. Gets called during __enter__."""
        pass

    def UpdateHeaderMessage(self, message):
        """Updates the header messsage if supported."""
        pass

    @abc.abstractmethod
    def Tick(self):
        """Give a visual indication to the user that some progress has been made.

    Output is sent to sys.stderr. Nothing is shown if output is not a TTY.

    Returns:
      Whether progress has completed.
    """
        pass

    def _GetTickMark(self, ticks):
        """Returns the next tick mark."""
        return self._symbols.spin_marks[self._ticks % len(self._symbols.spin_marks)]

    def _GetStagedCompletedSuffix(self, status):
        return status.value

    def _ValidateStage(self, key, allow_complete=False):
        """Validates the stage belongs to the tracker.

    Args:
      key: the key of the stage to validate.
      allow_complete: whether to error on an already-complete stage

    Returns:
      The validated Stage object, even if we were passed a key.
    """
        if key not in self:
            raise ValueError('This stage does not belong to this progress tracker.')
        stage = self.get(key)
        if not allow_complete and stage.status not in {StageCompletionStatus.NOT_STARTED, StageCompletionStatus.RUNNING}:
            raise ValueError('This stage has already completed.')
        return stage

    def StartStage(self, key):
        """Informs the progress tracker that this stage has started."""
        stage = self._ValidateStage(key)
        with self._lock:
            self._running_stages.add(key)
            stage.status = StageCompletionStatus.RUNNING
            self._StartStage(stage)
        self.Tick()

    def _StartStage(self, stage):
        """Override to customize behavior on starting a stage."""
        return

    def _FailStage(self, stage, failure_exception, message):
        """Override to customize behavior on failing a stage."""
        pass

    def _PrintExitOutput(self, aborted=False, warned=False, failed=False):
        """Override to customize behavior on printing exit output."""
        pass

    def UpdateStage(self, key, message):
        """Updates a stage in the progress tracker."""
        stage = self._ValidateStage(key)
        with self._lock:
            stage.message = message
        self.Tick()

    def CompleteStage(self, key, message=None):
        """Informs the progress tracker that this stage has completed."""
        stage = self._ValidateStage(key)
        with self._lock:
            stage.status = StageCompletionStatus.SUCCESS
            stage._is_done = True
            self._running_stages.discard(key)
            if message is not None:
                stage.message = message
            self._CompleteStage(stage)
        self.Tick()

    def _CompleteStage(self, stage):
        return

    def CompleteStageWithWarning(self, key, warning_message):
        self.CompleteStageWithWarnings(key, [warning_message])

    def CompleteStageWithWarnings(self, key, warning_messages):
        """Informs the progress tracker that this stage completed with warnings.

    Args:
      key: str, key for the stage to fail.
      warning_messages: list of str, user visible warning messages.
    """
        stage = self._ValidateStage(key)
        with self._lock:
            stage.status = StageCompletionStatus.WARNING
            stage._is_done = True
            self._running_stages.discard(key)
            self._exit_output_warnings.extend(warning_messages)
            self._completed_with_warnings_stages.append(stage.key)
            self._CompleteStageWithWarnings(stage, warning_messages)
        self.Tick()

    def _CompleteStageWithWarnings(self, stage, warning_messages):
        """Override to customize behavior on completing a stage with warnings."""
        pass

    def FailStage(self, key, failure_exception, message=None):
        """Informs the progress tracker that this stage has failed.

    Args:
      key: str, key for the stage to fail.
      failure_exception: Exception, raised by __exit__.
      message: str, user visible message for failure.
    """
        stage = self._ValidateStage(key)
        with self._lock:
            stage.status = StageCompletionStatus.FAILED
            stage._is_done = True
            self._running_stages.discard(key)
            if message is not None:
                stage.message = message
            self._FailStage(stage, failure_exception, message)
        self.Tick()
        if failure_exception:
            self._PrintExitOutput(failed=True)
            self._exception_is_uncaught = False
            raise failure_exception

    def AddWarning(self, warning_message):
        """Add a warning message independent of any particular stage.

    This warning message will be printed on __exit__.

    Args:
      warning_message: str, user visible warning message.
    """
        with self._lock:
            self._exit_output_warnings.append(warning_message)