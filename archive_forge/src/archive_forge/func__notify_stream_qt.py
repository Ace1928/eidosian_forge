import os
import platform
import sys
from functools import partial
import zmq
from packaging.version import Version as V
from traitlets.config.application import Application
def _notify_stream_qt(kernel):
    import operator
    from functools import lru_cache
    from IPython.external.qt_for_kernel import QtCore
    try:
        from IPython.external.qt_for_kernel import enum_helper
    except ImportError:

        @lru_cache(None)
        def enum_helper(name):
            return operator.attrgetter(name.rpartition('.')[0])(sys.modules[QtCore.__package__])

    def exit_loop():
        """fall back to main loop"""
        kernel._qt_notifier.setEnabled(False)
        kernel.app.qt_event_loop.quit()

    def process_stream_events():
        """fall back to main loop when there's a socket event"""
        if kernel.shell_stream.flush(limit=1):
            exit_loop()
    if not hasattr(kernel, '_qt_notifier'):
        fd = kernel.shell_stream.getsockopt(zmq.FD)
        kernel._qt_notifier = QtCore.QSocketNotifier(fd, enum_helper('QtCore.QSocketNotifier.Type').Read, kernel.app.qt_event_loop)
        kernel._qt_notifier.activated.connect(process_stream_events)
    else:
        kernel._qt_notifier.setEnabled(True)

    def _schedule_exit(delay):
        """schedule fall back to main loop in [delay] seconds"""
        if not hasattr(kernel, '_qt_timer'):
            kernel._qt_timer = QtCore.QTimer(kernel.app)
            kernel._qt_timer.setSingleShot(True)
            kernel._qt_timer.setTimerType(enum_helper('QtCore.Qt.TimerType').PreciseTimer)
            kernel._qt_timer.timeout.connect(exit_loop)
        kernel._qt_timer.start(int(1000 * delay))
    loop_qt._schedule_exit = _schedule_exit
    QtCore.QTimer.singleShot(0, process_stream_events)