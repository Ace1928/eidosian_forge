from typing import Dict, List, Sequence, Tuple, Optional
from vllm.block import BlockTable
Manages all the prompt prefixes.

    NOTE: This feature is experimental and may be replaced with automatic
        prefix caching in the future.

    Args:
        block_size: The block size of the executed model.

    Attributes:
        prefixes: A list of all the prefixes.
        block_size: The block size of the executed model.
    