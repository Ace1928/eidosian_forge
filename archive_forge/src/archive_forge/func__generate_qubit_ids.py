from typing import Callable, Dict, Optional, Sequence, Set, Tuple, Union, TYPE_CHECKING
import re
import numpy as np
from cirq import ops, linalg, protocols, value
def _generate_qubit_ids(self) -> Dict['cirq.Qid', str]:
    return {qubit: f'q[{i}]' for i, qubit in enumerate(self.qubits)}