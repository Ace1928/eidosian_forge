from pyudev.device import Device
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