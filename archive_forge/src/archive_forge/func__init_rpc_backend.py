from datetime import timedelta
import logging
import os
import threading
import warnings
from typing import Generator, Tuple
from urllib.parse import urlparse
import torch
import torch.distributed as dist
def _init_rpc_backend(backend=BackendType.TENSORPIPE, store=None, name=None, rank=-1, world_size=None, rpc_backend_options=None):
    _validate_rpc_args(backend, store, name, rank, world_size, rpc_backend_options)
    if _is_current_rpc_agent_set():
        raise RuntimeError('RPC is already initialized')
    rpc_agent = backend_registry.init_backend(backend, store=store, name=name, rank=rank, world_size=world_size, rpc_backend_options=rpc_backend_options)
    api._init_rpc_states(rpc_agent)