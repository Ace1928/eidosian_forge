from __future__ import annotations
import copy
from abc import abstractmethod
import numpy as np
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
from qiskit.result.counts import Counts
@staticmethod
def _matrix_to_dict(mat, dims, decimals=None, string_labels=False):
    """Convert a matrix to a ket dictionary.

        This representation will not show zero values in the output dict.

        Args:
            mat (array): a Numpy matrix array.
            dims (tuple): subsystem dimensions.
            decimals (None or int): number of decimal places to round to.
                                    (See Numpy.round), if None no rounding
                                    is done (Default: None).
            string_labels (bool): return ket as string if True, otherwise
                                  return as index array (Default: False).

        Returns:
            dict: the matrix in dictionary `ket` form.
        """
    vals = mat if decimals is None else mat.round(decimals=decimals)
    inds_row, inds_col = vals.nonzero()
    bras = QuantumState._index_to_ket_array(inds_row, dims, string_labels=string_labels)
    kets = QuantumState._index_to_ket_array(inds_col, dims, string_labels=string_labels)
    if string_labels:
        return {f'{ket}|{bra}': val for ket, bra, val in zip(kets, bras, vals[inds_row, inds_col])}
    return {(tuple(ket), tuple(bra)): val for ket, bra, val in zip(kets, bras, vals[inds_row, inds_col])}