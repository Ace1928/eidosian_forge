import importlib
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Tuple
from ._base import FILENAME_PATTERN, MAX_SHARD_SIZE, StateDictSplit, split_state_dict_into_shards_factory

    Taken from https://github.com/huggingface/safetensors/blob/08db34094e9e59e2f9218f2df133b7b4aaff5a99/bindings/python/py_src/safetensors/torch.py#L344
    