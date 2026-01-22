from gi.repository import GLib, GObject  # pylint: disable=import-error
class _ObserverMixin(object):
    """Mixin to provide observer behavior to the old and the new API."""

    def _setup_observer(self, monitor):
        self.monitor = monitor
        self.event_source = None
        self.enabled = True

    @property
    def enabled(self):
        """
        Whether this observer is enabled or not.

        If ``True`` (the default), this observer is enabled, and emits events.
        Otherwise it is disabled and does not emit any events.

        .. versionadded:: 0.14
        """
        return self.event_source is not None

    @enabled.setter
    def enabled(self, value):
        if value and self.event_source is None:
            self.event_source = GLib.io_add_watch(self.monitor, GLib.PRIORITY_DEFAULT, GLib.IO_IN, self._process_udev_event)
        elif not value and self.event_source is not None:
            GLib.source_remove(self.event_source)

    def _process_udev_event(self, source, condition):
        if condition == GLib.IO_IN:
            device = self.monitor.poll(timeout=0)
            if device is not None:
                self._emit_event(device)
        return True

    def _emit_event(self, device):
        self.emit('device-event', device)