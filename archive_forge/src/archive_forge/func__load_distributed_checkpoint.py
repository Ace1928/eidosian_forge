import os
import pickle
import warnings
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Optional, OrderedDict, Sequence, Set, Tuple, Union
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torch._C import _TensorMeta
from torch.nn import Parameter
from typing_extensions import override
from lightning_fabric.utilities.imports import (
from lightning_fabric.utilities.types import _PATH, _Stateful
def _load_distributed_checkpoint(checkpoint_folder: Path) -> Dict[str, Any]:
    """Loads a sharded checkpoint saved with the `torch.distributed.checkpoint` into a full state dict.

    The current implementation assumes that the entire checkpoint fits in CPU memory.

    """
    if not _TORCH_GREATER_EQUAL_2_1:
        raise ImportError('Processing distributed checkpoints requires PyTorch >= 2.1.')
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import BytesStorageMetadata, TensorStorageMetadata
    if _TORCH_GREATER_EQUAL_2_2:
        from torch.distributed.checkpoint import load
    else:
        from torch.distributed.checkpoint import load_state_dict as load
    reader = FileSystemReader(checkpoint_folder)
    metadata = reader.read_metadata()
    checkpoint: Dict[str, Any] = {}
    for tensor_name, sd_metadata in metadata.state_dict_metadata.items():
        if isinstance(sd_metadata, BytesStorageMetadata):
            checkpoint[tensor_name] = '<bytes_io>'
        elif isinstance(sd_metadata, TensorStorageMetadata):
            checkpoint[tensor_name] = torch.empty(size=sd_metadata.size, dtype=sd_metadata.properties.dtype, device=torch.device('cpu'), memory_format=sd_metadata.properties.memory_format, layout=sd_metadata.properties.layout, requires_grad=sd_metadata.properties.requires_grad, pin_memory=sd_metadata.properties.pin_memory)
    load(state_dict=checkpoint, storage_reader=reader, no_dist=True)
    checkpoint = _unflatten_dict(checkpoint, key_map=metadata.planner_data)
    extra_file = checkpoint_folder / _METADATA_FILENAME
    extra = torch.load(extra_file, map_location='cpu') if extra_file.is_file() else {}
    checkpoint.update(extra)
    return checkpoint