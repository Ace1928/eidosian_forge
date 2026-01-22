from typing import Dict, List, Union, Optional, Any
from typing_extensions import Literal
from pydantic import BaseModel, Field
from rpcq.messages import TargetDevice as TargetQuantumProcessor
def add_qubit(quantum_processor: CompilerISA, node_id: int) -> Qubit:
    if node_id not in quantum_processor.qubits:
        quantum_processor.qubits[str(node_id)] = Qubit(id=node_id)
    return quantum_processor.qubits[str(node_id)]