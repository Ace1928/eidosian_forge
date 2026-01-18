from __future__ import annotations
import functools
import logging
from collections import defaultdict
from typing import (
from langsmith import run_helpers
def wrap_openai(client: C) -> C:
    """Patch the OpenAI client to make it traceable.

    Args:
        client (Union[OpenAI, AsyncOpenAI]): The client to patch.

    Returns:
        Union[OpenAI, AsyncOpenAI]: The patched client.

    """
    client.chat.completions.create = _get_wrapper(client.chat.completions.create, 'ChatOpenAI', _reduce_chat)
    client.completions.create = _get_wrapper(client.completions.create, 'OpenAI', _reduce_completions)
    return client