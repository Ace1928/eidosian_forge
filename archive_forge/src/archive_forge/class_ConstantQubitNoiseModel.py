from typing import Any, Dict, Iterable, Sequence, TYPE_CHECKING, Union, Callable
from cirq import ops, protocols, value
from cirq._import import LazyLoader
from cirq._doc import document
@value.value_equality
class ConstantQubitNoiseModel(NoiseModel):
    """Applies noise to each qubit individually at the start of every moment.

    This is the noise model that is wrapped around an operation when that
    operation is given as "the noise to use" for a `NOISE_MODEL_LIKE` parameter.
    """

    def __init__(self, qubit_noise_gate: 'cirq.Gate', prepend: bool=False):
        """Noise model which applies a specific gate as noise to all gates.

        Args:
            qubit_noise_gate: The "noise" gate to use.
            prepend: If True, put noise before affected gates. Default: False.

        Raises:
            ValueError: if qubit_noise_gate is not a single-qubit gate.
        """
        if qubit_noise_gate.num_qubits() != 1:
            raise ValueError('noise.num_qubits() != 1')
        self.qubit_noise_gate = qubit_noise_gate
        self._prepend = prepend

    def _value_equality_values_(self) -> Any:
        return self.qubit_noise_gate

    def __repr__(self) -> str:
        return f'cirq.ConstantQubitNoiseModel({self.qubit_noise_gate!r})'

    def noisy_moment(self, moment: 'cirq.Moment', system_qubits: Sequence['cirq.Qid']):
        if self.is_virtual_moment(moment):
            return moment
        output = [moment, moment_module.Moment([self.qubit_noise_gate(q).with_tags(ops.VirtualTag()) for q in system_qubits])]
        return output[::-1] if self._prepend else output

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['qubit_noise_gate'])

    def _has_unitary_(self) -> bool:
        return protocols.has_unitary(self.qubit_noise_gate)

    def _has_mixture_(self) -> bool:
        return protocols.has_mixture(self.qubit_noise_gate)