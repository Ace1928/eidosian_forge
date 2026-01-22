from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import attr
import cirq
from cirq._compat import cached_property
from numpy.typing import NDArray
from cirq_ft import infra
from cirq_ft.algos import and_gate
from cirq_ft.deprecation import deprecated_cirq_ft_class
@deprecated_cirq_ft_class()
@attr.frozen
class ContiguousRegisterGate(cirq.ArithmeticGate):
    """Applies U|x>|y>|0> -> |x>|y>|x(x-1)/2 + y>

    This is useful in the case when $|x>$ and $|y>$ represent two selection signature such that
     $y < x$. For example, imagine a classical for-loop over two variables $x$ and $y$:

    >>> N = 10
    >>> data = [[1000 * x +  10 * y for y in range(x)] for x in range(N)]
    >>> for x in range(N):
    ...     for y in range(x):
    ...         # Iterates over a total of (N * (N - 1)) / 2 elements.
    ...         assert data[x][y] == 1000 * x + 10 * y

    We can rewrite the above using a single for-loop that uses a "contiguous" variable `i` s.t.

    >>> import numpy as np
    >>> N = 10
    >>> data = [[1000 * x + 10 * y for y in range(x)] for x in range(N)]
    >>> for i in range((N * (N - 1)) // 2):
    ...     x = int(np.floor((1 + np.sqrt(1 + 8 * i)) / 2))
    ...     y = i - (x * (x - 1)) // 2
    ...     assert data[x][y] == 1000 * x + 10 * y

     Note that both the for-loops iterate over the same ranges and in the same order. The only
     difference is that the second loop is a "flattened" version of the first one.

     Such a flattening of selection signature is useful when we want to load multi dimensional
     data to a target register which is indexed on selection signature $x$ and $y$ such that
     $0<= y <= x < N$ and we want to use a `SelectSwapQROM` to laod this data; which gives a
     sqrt-speedup over a traditional QROM at the cost of using more memory and loading chunks
     of size `sqrt(N)` in a single iteration. See the reference for more details.

     References:
         [Even More Efficient Quantum Computations of Chemistry Through Tensor Hypercontraction]
         (https://arxiv.org/abs/2011.03494)
            Lee et. al. (2020). Appendix F, Page 67.
    """
    bitsize: int
    target_bitsize: int

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return ([2] * self.bitsize, [2] * self.bitsize, [2] * self.target_bitsize)

    def with_registers(self, *new_registers) -> 'ContiguousRegisterGate':
        x_bitsize, y_bitsize, target_bitsize = [len(reg) for reg in new_registers]
        assert x_bitsize == y_bitsize, f'x_bitsize={x_bitsize} should be same as y_bitsize={y_bitsize}'
        return ContiguousRegisterGate(x_bitsize, target_bitsize)

    def apply(self, *register_vals: int) -> Union[int, Iterable[int]]:
        x, y, target = register_vals
        return (x, y, target ^ x * (x - 1) // 2 + y)

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ['In(x)'] * self.bitsize
        wire_symbols += ['In(y)'] * self.bitsize
        wire_symbols += ['+(x(x-1)/2 + y)'] * self.target_bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def _t_complexity_(self) -> infra.TComplexity:
        toffoli_complexity = infra.t_complexity(cirq.CCNOT)
        return (self.bitsize ** 2 + self.bitsize - 1) * toffoli_complexity

    def __repr__(self) -> str:
        return f'cirq_ft.ContiguousRegisterGate({self.bitsize}, {self.target_bitsize})'

    def __pow__(self, power: int):
        if power in [1, -1]:
            return self
        return NotImplemented