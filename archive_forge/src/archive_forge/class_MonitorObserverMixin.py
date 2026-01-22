from pyudev.device import Device
class MonitorObserverMixin(object):
    """
    Base mixin for pyqt monitor observers.
    """

    def _setup_notifier(self, monitor, notifier_class):
        self.monitor = monitor
        self.notifier = notifier_class(monitor.fileno(), notifier_class.Read, self)
        self.notifier.activated[int].connect(self._process_udev_event)

    @property
    def enabled(self):
        """
        Whether this observer is enabled or not.

        If ``True`` (the default), this observer is enabled, and emits events.
        Otherwise it is disabled and does not emit any events.  This merely
        reflects the state of the ``enabled`` property of the underlying
        :attr:`notifier`.

        .. versionadded:: 0.14
        """
        return self.notifier.isEnabled()

    @enabled.setter
    def enabled(self, value):
        self.notifier.setEnabled(value)

    def _process_udev_event(self):
        """
        Attempt to receive a single device event from the monitor, process
        the event and emit corresponding signals.

        Called by ``QSocketNotifier``, if data is available on the udev
        monitoring socket.
        """
        device = self.monitor.poll(timeout=0)
        if device is not None:
            self._emit_event(device)

    def _emit_event(self, device):
        self.deviceEvent.emit(device)