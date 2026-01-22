import json
import logging
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union
from ray._private import ray_option_utils
from ray.util.client.runtime_context import _ClientWorkerPropertyAPI
def _internal_kv_initialized(self) -> bool:
    """Hook for internal_kv._internal_kv_initialized."""
    return True