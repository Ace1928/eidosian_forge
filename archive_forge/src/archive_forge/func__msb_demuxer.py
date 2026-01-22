from typing import List, Callable, TYPE_CHECKING
from scipy.linalg import cossin
import numpy as np
from cirq import ops
from cirq.linalg import decompositions, predicates
def _msb_demuxer(demux_qubits: 'List[cirq.Qid]', u1: np.ndarray, u2: np.ndarray) -> 'op_tree.OpTree':
    """Demultiplexes a unitary matrix that is multiplexed in its most-significant-qubit.

    Decomposition structure:
      [ u1 , 0  ]  =  [ V , 0 ][ D , 0  ][ W , 0 ]
      [ 0  , u2 ]     [ 0 , V ][ 0 , D* ][ 0 , W ]

      Gives: ( u1 )( u2* ) = ( V )( D^2 )( V* )
         and:  W = ( D )( V* )( u2 )

    Args:
        demux_qubits: Subset of total qubits involved in this unitary gate
        u1: Upper-left quadrant of total unitary to be decomposed (see diagram)
        u2: Lower-right quadrant of total unitary to be decomposed (see diagram)

    Calls:
        1. quantum_shannon_decomposition
        2. _multiplexed_cossin
        3. quantum_shannon_decomposition

    Yields: Single operation from OP TREE of 2-qubit and 1-qubit operations
    """
    u = u1 @ u2.T.conjugate()
    dsquared, V = np.linalg.eig(u)
    d = np.sqrt(dsquared)
    D = np.diag(d)
    W = D @ V.T.conjugate() @ u2
    yield from quantum_shannon_decomposition(demux_qubits[1:], W)
    yield from _multiplexed_cossin(demux_qubits, -np.angle(d), ops.rz)
    yield from quantum_shannon_decomposition(demux_qubits[1:], V)