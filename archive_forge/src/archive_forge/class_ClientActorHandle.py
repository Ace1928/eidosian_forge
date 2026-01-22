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
class ClientActorHandle(ClientStub):
    """Client-side stub for instantiated actor.

    A stub created on the Ray Client to represent a remote actor that
    has been started on the cluster.  This class is allowed to be passed
    around between remote functions.

    Args:
        actor_ref: A reference to the running actor given to the client. This
          is a serialized version of the actual handle as an opaque token.
    """

    def __init__(self, actor_ref: ClientActorRef, actor_class: Optional[ClientActorClass]=None):
        self.actor_ref = actor_ref
        self._dir: Optional[List[str]] = None
        if actor_class is not None:
            self._method_num_returns = {}
            self._method_signatures = {}
            for method_name, method_obj in inspect.getmembers(actor_class.actor_cls, is_function_or_method):
                self._method_num_returns[method_name] = getattr(method_obj, '__ray_num_returns__', None)
                self._method_signatures[method_name] = inspect.Signature(parameters=extract_signature(method_obj, ignore_first=not (is_class_method(method_obj) or is_static_method(actor_class.actor_cls, method_name))))
        else:
            self._method_num_returns = None
            self._method_signatures = None

    def __dir__(self) -> List[str]:
        if self._method_num_returns is not None:
            return self._method_num_returns.keys()
        if ray.is_connected():
            self._init_class_info()
            return self._method_num_returns.keys()
        return super().__dir__()

    @property
    def _actor_id(self) -> ClientActorRef:
        return self.actor_ref

    def __getattr__(self, key):
        if key == '_method_num_returns':
            raise AttributeError(f"ClientActorRef has no attribute '{key}'")
        if self._method_num_returns is None:
            self._init_class_info()
        if key not in self._method_signatures:
            raise AttributeError(f"ClientActorRef has no attribute '{key}'")
        return ClientRemoteMethod(self, key, self._method_num_returns.get(key), self._method_signatures.get(key))

    def __repr__(self):
        return 'ClientActorHandle(%s)' % self.actor_ref.id.hex()

    def _init_class_info(self):

        @ray.remote(num_cpus=0)
        def get_class_info(x):
            return (x._ray_method_num_returns, x._ray_method_signatures)
        self._method_num_returns, method_parameters = ray.get(get_class_info.remote(self))
        self._method_signatures = {}
        for method, parameters in method_parameters.items():
            self._method_signatures[method] = inspect.Signature(parameters=parameters)