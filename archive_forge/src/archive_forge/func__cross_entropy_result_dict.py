import datetime
import functools
from typing import Dict, List, NamedTuple, Optional, Tuple, TYPE_CHECKING
from cirq.protocols.json_serialization import ObjectFactory
def _cross_entropy_result_dict(results: List[Tuple[List['cirq.Qid'], CrossEntropyResult]], **kwargs) -> CrossEntropyResultDict:
    return CrossEntropyResultDict(results={tuple(qubits): result for qubits, result in results})