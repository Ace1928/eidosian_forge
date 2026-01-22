import collections
import logging
import threading
from typing import Callable, Optional, Type
import grpc
from grpc import _common
from grpc._cython import cygrpc
from grpc._typing import MetadataType
class _AuthMetadataPluginCallback(grpc.AuthMetadataPluginCallback):
    _state: _CallbackState
    _callback: Callable

    def __init__(self, state: _CallbackState, callback: Callable):
        self._state = state
        self._callback = callback

    def __call__(self, metadata: MetadataType, error: Optional[Type[BaseException]]):
        with self._state.lock:
            if self._state.exception is None:
                if self._state.called:
                    raise RuntimeError('AuthMetadataPluginCallback invoked more than once!')
                else:
                    self._state.called = True
            else:
                raise RuntimeError('AuthMetadataPluginCallback raised exception "{}"!'.format(self._state.exception))
        if error is None:
            self._callback(metadata, cygrpc.StatusCode.ok, None)
        else:
            self._callback(None, cygrpc.StatusCode.internal, _common.encode(str(error)))