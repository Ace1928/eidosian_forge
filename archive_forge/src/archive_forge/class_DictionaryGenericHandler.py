import collections
import logging
import threading
import time
from typing import Callable, Dict, Optional, Sequence
import grpc  # pytype: disable=pyi-error
from grpc import _common  # pytype: disable=pyi-error
from grpc._typing import DoneCallbackType
class DictionaryGenericHandler(grpc.ServiceRpcHandler):
    _name: str
    _method_handlers: Dict[str, grpc.RpcMethodHandler]

    def __init__(self, service: str, method_handlers: Dict[str, grpc.RpcMethodHandler]):
        self._name = service
        self._method_handlers = {_common.fully_qualified_method(service, method): method_handler for method, method_handler in method_handlers.items()}

    def service_name(self) -> str:
        return self._name

    def service(self, handler_call_details: grpc.HandlerCallDetails) -> Optional[grpc.RpcMethodHandler]:
        details_method = handler_call_details.method
        return self._method_handlers.get(details_method)