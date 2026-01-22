import inspect
import logging
import os
import pickle
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import grpc
import ray._raylet as raylet
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray._private import ray_constants
from ray._private.inspect_util import (
from ray._private.signature import extract_signature, get_signature
from ray._private.utils import check_oversized_function
from ray.util.client import ray
from ray.util.client.options import validate_options
class ClientObjectRef(raylet.ObjectRef):

    def __init__(self, id: Union[bytes, Future]):
        self._mutex = threading.Lock()
        self._worker = ray.get_context().client_worker
        self._id_future = None
        if isinstance(id, bytes):
            self._set_id(id)
        elif isinstance(id, Future):
            self._id_future = id
        else:
            raise TypeError('Unexpected type for id {}'.format(id))

    def __del__(self):
        if self._worker is not None and self._worker.is_connected():
            try:
                if not self.is_nil():
                    self._worker.call_release(self.id)
            except Exception:
                logger.info('Exception in ObjectRef is ignored in destructor. To receive this exception in application code, call a method on the actor reference before its destructor is run.')

    def binary(self):
        self._wait_for_id()
        return super().binary()

    def hex(self):
        self._wait_for_id()
        return super().hex()

    def is_nil(self):
        self._wait_for_id()
        return super().is_nil()

    def __hash__(self):
        self._wait_for_id()
        return hash(self.id)

    def task_id(self):
        self._wait_for_id()
        return super().task_id()

    @property
    def id(self):
        return self.binary()

    def future(self) -> Future:
        fut = Future()

        def set_future(data: Any) -> None:
            """Schedules a callback to set the exception or result
            in the Future."""
            if isinstance(data, Exception):
                fut.set_exception(data)
            else:
                fut.set_result(data)
        self._on_completed(set_future)
        fut.object_ref = self
        return fut

    def _on_completed(self, py_callback: Callable[[Any], None]) -> None:
        """Register a callback that will be called after Object is ready.
        If the ObjectRef is already ready, the callback will be called soon.
        The callback should take the result as the only argument. The result
        can be an exception object in case of task error.
        """

        def deserialize_obj(resp: Union[ray_client_pb2.DataResponse, Exception]) -> None:
            from ray.util.client.client_pickler import loads_from_server
            if isinstance(resp, Exception):
                data = resp
            elif isinstance(resp, bytearray):
                data = loads_from_server(resp)
            else:
                obj = resp.get
                data = None
                if not obj.valid:
                    data = loads_from_server(resp.get.error)
                else:
                    data = loads_from_server(resp.get.data)
            py_callback(data)
        self._worker.register_callback(self, deserialize_obj)

    def _set_id(self, id):
        super()._set_id(id)
        self._worker.call_retain(id)

    def _wait_for_id(self, timeout=None):
        if self._id_future:
            with self._mutex:
                if self._id_future:
                    self._set_id(self._id_future.result(timeout=timeout))
                    self._id_future = None