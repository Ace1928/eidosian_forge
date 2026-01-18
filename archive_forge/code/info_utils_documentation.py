import enum
import os
from typing import Optional
from huggingface_hub.utils import insecure_hashlib
from .. import config
from .logging import get_logger
Check if `dataset_size` is smaller than `config.IN_MEMORY_MAX_SIZE`.

    Args:
        dataset_size (int): Dataset size in bytes.

    Returns:
        bool: Whether `dataset_size` is smaller than `config.IN_MEMORY_MAX_SIZE`.
    