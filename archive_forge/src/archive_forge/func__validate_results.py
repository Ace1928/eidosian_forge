import asyncio
import time
from dataclasses import dataclass
from functools import wraps
from inspect import isasyncgenfunction, iscoroutinefunction
from typing import (
from ray._private.signature import extract_signature, flatten_args, recover_args
from ray._private.utils import get_or_create_event_loop
from ray.serve._private.utils import extract_self_if_method_call
from ray.serve.exceptions import RayServeException
from ray.util.annotations import PublicAPI
def _validate_results(self, results: Iterable[Any], input_batch_length: int) -> None:
    if len(results) != input_batch_length:
        raise RayServeException(f"Batched function doesn't preserve batch size. The input list has length {input_batch_length} but the returned list has length {len(results)}.")