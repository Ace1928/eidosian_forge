import sys
import traceback
from taskflow.listeners import base
class PrintingListener(base.DumpingListener):
    """Writes the task and flow notifications messages to stdout or stderr."""

    def __init__(self, engine, task_listen_for=base.DEFAULT_LISTEN_FOR, flow_listen_for=base.DEFAULT_LISTEN_FOR, retry_listen_for=base.DEFAULT_LISTEN_FOR, stderr=False):
        super(PrintingListener, self).__init__(engine, task_listen_for=task_listen_for, flow_listen_for=flow_listen_for, retry_listen_for=retry_listen_for)
        if stderr:
            self._file = sys.stderr
        else:
            self._file = sys.stdout

    def _dump(self, message, *args, **kwargs):
        print(message % args, file=self._file)
        exc_info = kwargs.get('exc_info')
        if exc_info is not None:
            traceback.print_exception(exc_info[0], exc_info[1], exc_info[2], file=self._file)