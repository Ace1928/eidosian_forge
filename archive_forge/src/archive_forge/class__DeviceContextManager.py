import functools
import threading
from contextlib import contextmanager
from .driver import driver, USE_NV_BINDING
class _DeviceContextManager(object):
    """
    Provides a context manager for executing in the context of the chosen
    device. The normal use of instances of this type is from
    ``numba.cuda.gpus``. For example, to execute on device 2::

       with numba.cuda.gpus[2]:
           d_a = numba.cuda.to_device(a)

    to copy the array *a* onto device 2, referred to by *d_a*.
    """

    def __init__(self, device):
        self._device = device

    def __getattr__(self, item):
        return getattr(self._device, item)

    def __enter__(self):
        _runtime.get_or_create_context(self._device.id)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._device.get_primary_context().pop()

    def __str__(self):
        return '<Managed Device {self.id}>'.format(self=self)