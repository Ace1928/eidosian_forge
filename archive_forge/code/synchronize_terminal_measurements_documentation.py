from typing import List, Optional, Set, Tuple, TYPE_CHECKING
from cirq import protocols, circuits
from cirq.transformers import transformer_api
Move measurements to the end of the circuit.

    Move all measurements in a circuit to the final moment, if it can accommodate them (without
    overlapping with other operations). If `after_other_operations` is true, then a new moment will
    be added to the end of the circuit containing all the measurements that should be brought
    forward.

    Args:
          circuit: Input circuit to transform.
          context: `cirq.TransformerContext` storing common configurable options for transformers.
          after_other_operations: Set by default. If the circuit's final moment contains
                non-measurement operations and this is set then a new empty moment is appended to
                the circuit before pushing measurements to the end.
    Returns:
          Copy of the transformed input circuit.
    