from typing import Any, Dict, List, Sequence, Type, Union
import cirq
def _json_dict_(self) -> Dict[str, Any]:
    return {'atol': self.atol, 'eject_paulis': self.eject_paulis, 'additional_gates': list(self.additional_gates)}