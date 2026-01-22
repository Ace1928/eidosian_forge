from typing import Callable, List, Optional, Tuple, Set, Any, TYPE_CHECKING
import numpy as np
import cirq
from cirq_google.line.placement import place_strategy, optimization
from cirq_google.line.placement.chip import above, right_of, chip_as_adjacency_list, EDGE
from cirq_google.line.placement.sequence import GridQubitLineTuple, LineSequence
def _force_edge_active(self, seqs: List[List[cirq.GridQubit]], edge: EDGE, sample_bool: Callable[[], bool]) -> List[List[cirq.GridQubit]]:
    """Move which forces given edge to appear on some sequence.

        Args:
          seqs: List of linear sequences covering chip.
          edge: Edge to be activated.
          sample_bool: Callable returning random bool.

        Returns:
          New list of linear sequences with given edge on some of the
          sequences.
        """
    n0, n1 = edge
    seqs = list(seqs)
    i0, j0 = index_2d(seqs, n0)
    i1, j1 = index_2d(seqs, n1)
    s0 = seqs[i0]
    s1 = seqs[i1]
    if i0 != i1:
        part = ([s0[:j0], s0[j0 + 1:]], [s1[:j1], s1[j1 + 1:]])
        del seqs[max(i0, i1)]
        del seqs[min(i0, i1)]
        c0 = 0 if not part[0][1] else 1 if not part[0][0] else sample_bool()
        if c0:
            part[0][c0].reverse()
        c1 = 0 if not part[1][1] else 1 if not part[1][0] else sample_bool()
        if not c1:
            part[1][c1].reverse()
        seqs.append(part[0][c0] + [n0, n1] + part[1][c1])
        other = [1, 0]
        seqs.append(part[0][other[c0]])
        seqs.append(part[1][other[c1]])
    else:
        if j0 > j1:
            j0, j1 = (j1, j0)
            n0, n1 = (n1, n0)
        head = s0[:j0]
        inner = s0[j0 + 1:j1]
        tail = s0[j1 + 1:]
        del seqs[i0]
        if sample_bool():
            if sample_bool():
                seqs.append(inner + [n1, n0] + head[::-1])
                seqs.append(tail)
            else:
                seqs.append(tail[::-1] + [n1, n0] + inner)
                seqs.append(head)
        else:
            seqs.append(head + [n0, n1] + tail)
            seqs.append(inner)
    return [e for e in seqs if e]