import logging
import time
from typing import Any, AnyStr, Callable, Optional, Union
import grpc
from grpc._cython import cygrpc
from grpc._typing import DeserializingFunction
from grpc._typing import SerializingFunction
def _wait_once(wait_fn: Callable[..., bool], timeout: float, spin_cb: Optional[Callable[[], None]]):
    wait_fn(timeout=timeout)
    if spin_cb is not None:
        spin_cb()