import string
from typing import TYPE_CHECKING, Union, Any, Tuple, TypeVar, Optional, Dict, Iterable
from typing_extensions import Protocol
from cirq import ops
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
class QasmArgs(string.Formatter):
    """Formatting Arguments for outputting QASM code."""

    def __init__(self, precision: int=10, version: str='2.0', qubit_id_map: Optional[Dict['cirq.Qid', str]]=None, meas_key_id_map: Optional[Dict[str, str]]=None) -> None:
        """Inits QasmArgs.

        Args:
            precision: The number of digits after the decimal to show for
                numbers in the qasm code.
            version: The QASM version to target. Objects may return different
                qasm depending on version.
            qubit_id_map: A dictionary mapping qubits to qreg QASM identifiers.
            meas_key_id_map: A dictionary mapping measurement keys to creg QASM
                identifiers.
        """
        self.precision = precision
        self.version = version
        self.qubit_id_map = {} if qubit_id_map is None else qubit_id_map
        self.meas_key_id_map = {} if meas_key_id_map is None else meas_key_id_map

    def format_field(self, value: Any, spec: str) -> str:
        """Method of string.Formatter that specifies the output of format()."""
        if isinstance(value, (float, int)):
            if isinstance(value, float):
                value = round(value, self.precision)
            if spec == 'half_turns':
                value = f'pi*{value}' if value != 0 else '0'
                spec = ''
        elif isinstance(value, ops.Qid):
            value = self.qubit_id_map[value]
        elif isinstance(value, str) and spec == 'meas':
            value = self.meas_key_id_map[value]
            spec = ''
        return super().format_field(value, spec)

    def validate_version(self, *supported_versions: str) -> None:
        if self.version not in supported_versions:
            raise ValueError(f'QASM version {self.version} output is not supported.')