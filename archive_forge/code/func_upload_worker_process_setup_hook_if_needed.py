import traceback
import logging
import base64
import os
from typing import Dict, Any, Callable, Union, Optional
import ray
import ray._private.ray_constants as ray_constants
from ray._private.storage import _load_class
import ray.cloudpickle as pickle
from ray.runtime_env import RuntimeEnv
def upload_worker_process_setup_hook_if_needed(runtime_env: Union[Dict[str, Any], RuntimeEnv], worker: 'ray.Worker') -> Union[Dict[str, Any], RuntimeEnv]:
    """Uploads the worker_process_setup_hook to GCS with a key.

    runtime_env["worker_process_setup_hook"] is converted to a decoded key
    that can load the worker setup hook function from GCS.
    i.e., you can use internalKV.Get(runtime_env["worker_process_setup_hook])
    to access the worker setup hook from GCS.

    Args:
        runtime_env: The runtime_env. The value will be modified
            when returned.
        worker: ray.worker instance.
        decoder: GCS requires the function key to be bytes. However,
            we cannot json serialize (which is required to serialize
            runtime env) the bytes. So the key should be decoded to
            a string. The given decoder is used to decode the function
            key.
    """
    setup_func = runtime_env.get('worker_process_setup_hook')
    if setup_func is None:
        return runtime_env
    if isinstance(setup_func, Callable):
        return export_setup_func_callable(runtime_env, setup_func, worker)
    elif isinstance(setup_func, str):
        return export_setup_func_module(runtime_env, setup_func)
    else:
        raise TypeError(f'worker_process_setup_hook must be a function, got {type(setup_func)}.')