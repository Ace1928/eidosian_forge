from typing import Sequence, TYPE_CHECKING
from cirq import circuits, devices, value, ops
from cirq.devices.noise_model import validate_all_measurements
class DepolarizingNoiseModel(devices.NoiseModel):
    """Applies depolarizing noise to each qubit individually at the end of
    every moment.

    If a circuit contains measurements, they must be in moments that don't
    also contain gates.
    """

    def __init__(self, depol_prob: float, prepend: bool=False):
        """A depolarizing noise model

        Args:
            depol_prob: Depolarizing probability.
            prepend: If True, put noise before affected gates. Default: False.
        """
        value.validate_probability(depol_prob, 'depol prob')
        self.qubit_noise_gate = ops.DepolarizingChannel(depol_prob)
        self._prepend = prepend

    def noisy_moment(self, moment: 'cirq.Moment', system_qubits: Sequence['cirq.Qid']):
        if validate_all_measurements(moment) or self.is_virtual_moment(moment):
            return moment
        output = [moment, circuits.Moment((self.qubit_noise_gate(q).with_tags(ops.VirtualTag()) for q in system_qubits))]
        return output[::-1] if self._prepend else output