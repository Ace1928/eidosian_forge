from typing import Iterable, List, Set, TYPE_CHECKING
from cirq.ops import named_qubit, qid_util, qubit_manager
def _allocate_qid(self, name: str, dim: int) -> 'cirq.Qid':
    return qid_util.q(name) if dim == 2 else named_qubit.NamedQid(name, dimension=dim)