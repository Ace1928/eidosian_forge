from __future__ import annotations
import functools
from lazyops.libs.pooler import ThreadPooler
from lazyops.libs.logging import logger
from typing import Any, Dict, Optional, TypeVar, Callable, TYPE_CHECKING
def _get_ph():
    """
            Gets the PostHog Client
            """
    nonlocal client
    if client is None:
        client = get_posthog_client()
    return client