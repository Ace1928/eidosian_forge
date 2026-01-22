from pyudev.device import Device
def _emit_event(self, device):
    self.deviceEvent.emit(device.action, device)
    signal = self._action_signal_map.get(device.action)
    if signal is not None:
        signal.emit(device)