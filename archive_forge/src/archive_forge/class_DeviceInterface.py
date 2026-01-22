import inspect
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, Union
import torch
from torch._streambase import _EventBase, _StreamBase
class DeviceInterface(metaclass=DeviceInterfaceMeta):
    """
    This is a simple device runtime interface for Inductor. It enables custom
    backends to be integrated with Inductor in a device-agnostic semantic.
    """

    class device:

        def __new__(cls, device: _device_t):
            raise NotImplementedError()

    class Worker:
        """
        Worker API to query device properties that will work in multi processing
        workers that cannot use the GPU APIs (due to processing fork() and
        initialization time issues). Properties are recorded in the main process
        before we fork the workers.
        """

        @staticmethod
        def set_device(device: int):
            raise NotImplementedError()

        @staticmethod
        def current_device() -> int:
            raise NotImplementedError()

        @staticmethod
        def get_device_properties(device: _device_t=None):
            raise NotImplementedError()

    @staticmethod
    def current_device():
        raise NotImplementedError()

    @staticmethod
    def set_device(device: _device_t):
        raise NotImplementedError()

    @staticmethod
    def device_count():
        raise NotImplementedError()

    @staticmethod
    def is_available() -> bool:
        raise NotImplementedError()

    @staticmethod
    def stream(stream: torch.Stream):
        raise NotImplementedError()

    @staticmethod
    def current_stream():
        raise NotImplementedError()

    @staticmethod
    def set_stream(stream: torch.Stream):
        raise NotImplementedError()

    @staticmethod
    def _set_stream_by_id(stream_id: int, device_index: int, device_type: int):
        raise NotImplementedError()

    @staticmethod
    def get_raw_stream():
        raise NotImplementedError()

    @staticmethod
    def synchronize(device: _device_t=None):
        raise NotImplementedError()

    @staticmethod
    def get_device_properties(device: _device_t=None):
        raise NotImplementedError()

    @staticmethod
    def get_compute_capability(device: _device_t=None):
        raise NotImplementedError()