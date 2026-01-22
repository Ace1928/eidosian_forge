from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import attr
import cirq
from cirq._compat import cached_property
from numpy.typing import NDArray
from cirq_ft import infra
from cirq_ft.algos import and_gate
from cirq_ft.deprecation import deprecated_cirq_ft_class
def _right_building_block(self, inp, out, anc, depth):
    if depth == 0:
        return
    else:
        yield cirq.CX(anc[depth - 1], anc[depth])
        yield and_gate.And(adjoint=True).on(inp[depth], out[depth], anc[depth])
        yield cirq.CX(anc[depth - 1], inp[depth])
        yield cirq.CX(inp[depth], out[depth])
        yield from self._right_building_block(inp, out, anc, depth - 1)