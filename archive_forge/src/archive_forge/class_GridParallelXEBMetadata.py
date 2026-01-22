import dataclasses
from typing import Optional, Sequence, TYPE_CHECKING
from cirq.experiments.random_quantum_circuit_generation import GridInteractionLayer
from cirq import protocols
@dataclasses.dataclass
class GridParallelXEBMetadata:
    """Metadata for a grid parallel XEB experiment.
    Attributes:
        data_collection_id: The data collection ID of the experiment.
    """
    qubits: Sequence['cirq.Qid']
    two_qubit_gate: 'cirq.Gate'
    num_circuits: int
    repetitions: int
    cycles: Sequence[int]
    layers: Sequence[GridInteractionLayer]
    seed: Optional[int]

    def _json_dict_(self):
        return protocols.dataclass_json_dict(self)

    def __repr__(self) -> str:
        return f'cirq.experiments.grid_parallel_two_qubit_xeb.GridParallelXEBMetadata(qubits={self.qubits!r}, two_qubit_gate={self.two_qubit_gate!r}, num_circuits={self.num_circuits!r}, repetitions={self.repetitions!r}, cycles={self.cycles!r}, layers={self.layers!r}, seed={self.seed!r})'