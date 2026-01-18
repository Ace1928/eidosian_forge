from typing import Sequence, Union, Tuple
from numpy.typing import NDArray
import attr
import cirq
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import multi_control_multi_target_pauli as mcmtp
TComplexity as explained in Appendix B.2.c of https://arxiv.org/abs/1812.00954