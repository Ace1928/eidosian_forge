from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
class _FakeOperation(object):
    """A fake Operation object to pass to device functions."""

    def __init__(self):
        self.device = ''
        self.type = ''
        self.name = ''
        self.node_def = _FakeNodeDef()

    def _set_device(self, device):
        self.device = ops._device_string(device)

    def _set_device_from_string(self, device_str):
        self.device = device_str