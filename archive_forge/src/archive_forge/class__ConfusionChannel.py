import itertools
from collections import defaultdict
from typing import Any, cast, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import linalg, ops, protocols, value
from cirq.linalg import transformations
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.synchronize_terminal_measurements import find_terminal_measurements
class _ConfusionChannel(ops.Gate):
    """The quantum equivalent of a confusion matrix.

    This gate performs a complete dephasing of the input qubits, and then confuses the remaining
    diagonal components per the input confusion matrix.

    For a classical confusion matrix, the quantum equivalent is a channel that can be calculated
    by transposing the matrix, taking the square root of each term, and forming a Kraus sequence
    of each term individually and the rest zeroed out. For example, consider the confusion matrix

    $$
    \\begin{aligned}
    M_C =& \\begin{bmatrix}
               0.8 & 0.2  \\\\
               0.1 & 0.9
           \\end{bmatrix}
    \\end{aligned}
    $$

    If $a$ and $b (= 1-a)$ are probabilities of two possible classical states for a measurement,
    the confusion matrix operates on those probabilities as

    $$
    (a, b) M_C = (0.8a + 0.1b, 0.2a + 0.9b)
    $$

    This is equivalent to the following Kraus representation operating on a diagonal of a density
    matrix:

    $$
    \\begin{aligned}
    M_0 =& \\begin{bmatrix}
               \\sqrt{0.8} & 0  \\\\
               0 & 0
           \\end{bmatrix}
    \\\\
    M_1 =& \\begin{bmatrix}
               0 & \\sqrt{0.1} \\\\
               0 & 0
           \\end{bmatrix}
    \\\\
    M_2 =&  \\begin{bmatrix}
               0 & 0 \\\\
               \\sqrt{0.2} & 0
            \\end{bmatrix}
    \\\\
    M_3 =&  \\begin{bmatrix}
               0 & 0 \\\\
               0 & \\sqrt{0.9}
            \\end{bmatrix}
    \\end{aligned}
    \\\\
    $$
    Then for
    $$
    \\begin{aligned}
    \\rho =& \\begin{bmatrix}
               a & ?  \\\\
               ? & b
           \\end{bmatrix}
    \\end{aligned}
    \\\\
    \\\\
    $$
    the evolution of
    $$
    \\rho \\rightarrow M_0 \\rho M_0^\\dagger
                       + M_1 \\rho M_1^\\dagger
                       + M_2 \\rho M_2^\\dagger
                       + M_3 \\rho M_3^\\dagger
    $$
    gives the result
    $$
    \\begin{aligned}
    \\rho =& \\begin{bmatrix}
               0.8a + 0.1b & 0  \\\\
               0 & 0.2a + 0.9b
           \\end{bmatrix}
    \\end{aligned}
    \\\\
    $$

    Thus in a deferred measurement scenario, applying this channel to the ancilla qubit will model
    the noise distribution that would have been caused by the confusion matrix. The math
    generalizes cleanly to n-dimensional measurements as well.
    """

    def __init__(self, confusion_map: np.ndarray, shape: Sequence[int]):
        if confusion_map.ndim != 2:
            raise ValueError('Confusion map must be 2D.')
        row_count, col_count = confusion_map.shape
        if row_count != col_count:
            raise ValueError('Confusion map must be square.')
        if row_count != np.prod(shape):
            raise ValueError('Confusion map size does not match qubit shape.')
        kraus = []
        for r in range(row_count):
            for c in range(col_count):
                v = confusion_map[r, c]
                if v < 0:
                    raise ValueError('Confusion map has negative probabilities.')
                if v > 0:
                    m = np.zeros(confusion_map.shape)
                    m[c, r] = np.sqrt(v)
                    kraus.append(m)
        if not linalg.is_cptp(kraus_ops=kraus):
            raise ValueError('Confusion map has invalid probabilities.')
        self._shape = tuple(shape)
        self._confusion_map = confusion_map.copy()
        self._kraus = tuple(kraus)

    def _qid_shape_(self) -> Tuple[int, ...]:
        return self._shape

    def _kraus_(self) -> Tuple[np.ndarray, ...]:
        return self._kraus

    def _apply_channel_(self, args: 'cirq.ApplyChannelArgs'):
        configs: List[transformations._BuildFromSlicesArgs] = []
        for i in range(np.prod(self._shape) ** 2):
            scale = cast(complex, self._confusion_map.flat[i])
            if scale == 0:
                continue
            index: Any = np.unravel_index(i, self._shape * 2)
            slices: List[transformations._SliceConfig] = []
            axis_count = len(args.left_axes)
            for j in range(axis_count):
                s1 = transformations._SliceConfig(axis=args.left_axes[j], source_index=index[j], target_index=index[j + axis_count])
                s2 = transformations._SliceConfig(axis=args.right_axes[j], source_index=index[j], target_index=index[j + axis_count])
                slices.extend([s1, s2])
            configs.append(transformations._BuildFromSlicesArgs(slices=tuple(slices), scale=scale))
        transformations._build_from_slices(configs, args.target_tensor, out=args.out_buffer)
        return args.out_buffer