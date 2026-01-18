import warnings
from typing import Iterable, List, NamedTuple, Tuple, Union
import torch
from torch import Tensor
from ... import _VF
from ..._jit_internal import Optional
def unpack_sequence(packed_sequences: PackedSequence) -> List[Tensor]:
    """Unpack PackedSequence into a list of variable length Tensors.

    ``packed_sequences`` should be a PackedSequence object.


    Example:
        >>> from torch.nn.utils.rnn import pack_sequence, unpack_sequence
        >>> a = torch.tensor([1, 2, 3])
        >>> b = torch.tensor([4, 5])
        >>> c = torch.tensor([6])
        >>> sequences = [a, b, c]
        >>> print(sequences)
        [tensor([1, 2, 3]), tensor([4, 5]), tensor([6])]
        >>> packed_sequences = pack_sequence(sequences)
        >>> print(packed_sequences)
        PackedSequence(data=tensor([1, 4, 6, 2, 5, 3]), batch_sizes=tensor([3, 2, 1]), sorted_indices=None, unsorted_indices=None)
        >>> unpacked_sequences = unpack_sequence(packed_sequences)
        >>> print(unpacked_sequences)
        [tensor([1, 2, 3]), tensor([4, 5]), tensor([6])]


    Args:
        packed_sequences (PackedSequence): A PackedSequence object.

    Returns:
        a list of :class:`Tensor` objects
    """
    padded_sequences, lengths = pad_packed_sequence(packed_sequences, batch_first=True)
    unpacked_sequences = unpad_sequence(padded_sequences, lengths, batch_first=True)
    return unpacked_sequences