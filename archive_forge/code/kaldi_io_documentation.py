from typing import Any, Callable, Iterable, Tuple
import torch
from torch import Tensor
from torchaudio._internal import module_utils as _mod_utils
Create generator of (key,matrix<float32/float64>) tuples, which reads from the ark file/stream.

    Args:
        file_or_fd (str/FileDescriptor): ark, gzipped ark, pipe or opened file descriptor

    Returns:
        Iterable[Tuple[str, Tensor]]: The string is the key and the tensor is the matrix read from file

    Example
        >>> # read ark to a 'dictionary'
        >>> d = { u:d for u,d in torchaudio.kaldi_io.read_mat_ark(file) }
    