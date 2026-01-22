import functools
import operator
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
@dataclass
class SafetensorsFileMetadata:
    """Metadata for a Safetensors file hosted on the Hub.

    This class is returned by [`parse_safetensors_file_metadata`].

    For more details regarding the safetensors format, check out https://huggingface.co/docs/safetensors/index#format.

    Attributes:
        metadata (`Dict`):
            The metadata contained in the file.
        tensors (`Dict[str, TensorInfo]`):
            A map of all tensors. Keys are tensor names and values are information about the corresponding tensor, as a
            [`TensorInfo`] object.
        parameter_count (`Dict[str, int]`):
            A map of the number of parameters per data type. Keys are data types and values are the number of parameters
            of that data type.
    """
    metadata: Dict[str, str]
    tensors: Dict[TENSOR_NAME_T, TensorInfo]
    parameter_count: Dict[DTYPE_T, int] = field(init=False)

    def __post_init__(self) -> None:
        parameter_count: Dict[DTYPE_T, int] = defaultdict(int)
        for tensor in self.tensors.values():
            parameter_count[tensor.dtype] += tensor.parameter_count
        self.parameter_count = dict(parameter_count)