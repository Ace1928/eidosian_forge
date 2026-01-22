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
def _batch_args_kwargs(list_of_flattened_args: List[List[Any]]) -> Tuple[Tuple[Any], Dict[Any, Any]]:
    """Batch a list of flatten args and returns regular args and kwargs"""
    arg_lengths = {len(args) for args in list_of_flattened_args}
    assert len(arg_lengths) == 1, 'All batch requests should have the same number of parameters.'
    arg_length = arg_lengths.pop()
    batched_flattened_args = []
    for idx in range(arg_length):
        if idx % 2 == 0:
            batched_flattened_args.append(list_of_flattened_args[0][idx])
        else:
            batched_flattened_args.append([item[idx] for item in list_of_flattened_args])
    return recover_args(batched_flattened_args)