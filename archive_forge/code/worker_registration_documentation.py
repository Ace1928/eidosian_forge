from typing import TYPE_CHECKING, Optional, Set
from rq.utils import split_list
from .utils import as_text
Delete invalid worker keys in registry.

    Args:
        queue (Queue): The Queue
    