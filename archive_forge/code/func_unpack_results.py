import json
from typing import Any, cast, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Iterator
import numpy as np
import sympy
import cirq
from cirq_google.api.v1 import operations_pb2
def unpack_results(data: bytes, repetitions: int, key_sizes: Sequence[Tuple[str, int]]) -> Dict[str, np.ndarray]:
    """Unpack data from a bitstring into individual measurement results.

    Args:
        data: Packed measurement results, in the form <rep0><rep1>...
            where each repetition is <key0_0>..<key0_{size0-1}><key1_0>...
            with bits packed in little-endian order in each byte.
        repetitions: number of repetitions.
        key_sizes: Keys and sizes of the measurements in the data.

    Returns:
        Dict mapping measurement key to a 2D array of boolean results. Each
        array has shape (repetitions, size) with size for that measurement.
    """
    bits_per_rep = sum((size for _, size in key_sizes))
    total_bits = repetitions * bits_per_rep
    byte_arr = np.frombuffer(data, dtype='uint8').reshape((len(data), 1))
    bits = np.unpackbits(byte_arr, axis=1)[:, ::-1].reshape(-1).astype(bool)
    bits = bits[:total_bits].reshape((repetitions, bits_per_rep))
    results = {}
    ofs = 0
    for key, size in key_sizes:
        results[key] = bits[:, ofs:ofs + size]
        ofs += size
    return results