from __future__ import annotations
import asyncio
import copy
import os
import random
import time
import traceback
import uuid
from collections import defaultdict
from queue import Queue as ThreadQueue
from typing import TYPE_CHECKING
import fastapi
from typing_extensions import Literal
from gradio import route_utils, routes
from gradio.data_classes import (
from gradio.exceptions import Error
from gradio.helpers import TrackedIterable
from gradio.server_messages import (
from gradio.utils import LRUCache, run_coro_in_background, safe_get_lock, set_task_name
def _resolve_concurrency_limit(self, default_concurrency_limit: int | None | Literal['not_set']) -> int | None:
    """
        Handles the logic of resolving the default_concurrency_limit as this can be specified via a combination
        of the `default_concurrency_limit` parameter of the `Blocks.queue()` or the `GRADIO_DEFAULT_CONCURRENCY_LIMIT`
        environment variable. The parameter in `Blocks.queue()` takes precedence over the environment variable.
        Parameters:
            default_concurrency_limit: The default concurrency limit, as specified by a user in `Blocks.queu()`.
        """
    if default_concurrency_limit != 'not_set':
        return default_concurrency_limit
    if (default_concurrency_limit_env := os.environ.get('GRADIO_DEFAULT_CONCURRENCY_LIMIT')):
        if default_concurrency_limit_env.lower() == 'none':
            return None
        else:
            return int(default_concurrency_limit_env)
    else:
        return 1