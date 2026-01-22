from .Qt import QtCore
class ThreadsafeTimer(QtCore.QObject):
    """
    Thread-safe replacement for QTimer.
    """
    timeout = QtCore.Signal()
    sigTimerStopRequested = QtCore.Signal()
    sigTimerStartRequested = QtCore.Signal(object)

    def __init__(self):
        QtCore.QObject.__init__(self)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.timerFinished)
        self.timer.moveToThread(QtCore.QCoreApplication.instance().thread())
        self.moveToThread(QtCore.QCoreApplication.instance().thread())
        self.sigTimerStopRequested.connect(self.stop, QtCore.Qt.ConnectionType.QueuedConnection)
        self.sigTimerStartRequested.connect(self.start, QtCore.Qt.ConnectionType.QueuedConnection)

    def start(self, timeout):
        isGuiThread = QtCore.QThread.currentThread() == QtCore.QCoreApplication.instance().thread()
        if isGuiThread:
            self.timer.start(int(timeout))
        else:
            self.sigTimerStartRequested.emit(timeout)

    def stop(self):
        isGuiThread = QtCore.QThread.currentThread() == QtCore.QCoreApplication.instance().thread()
        if isGuiThread:
            self.timer.stop()
        else:
            self.sigTimerStopRequested.emit()

    def timerFinished(self):
        self.timeout.emit()