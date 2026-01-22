from typing import Sequence, Tuple
import numpy as np
import pennylane as qml
from pennylane.wires import Wires
from .measurements import Probability, SampleMeasurement, StateMeasurement
from .mid_measure import MeasurementValue
class ProbabilityMP(SampleMeasurement, StateMeasurement):
    """Measurement process that computes the probability of each computational basis state.

    Please refer to :func:`probs` for detailed documentation.

    Args:
        obs (Union[.Operator, .MeasurementValue]): The observable that is to be measured
            as part of the measurement process. Not all measurement processes require observables
            (for example ``Probability``); this argument is optional.
        wires (.Wires): The wires the measurement process applies to.
            This can only be specified if an observable was not provided.
        eigvals (array): A flat array representing the eigenvalues of the measurement.
            This can only be specified if an observable was not provided.
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
    """

    @property
    def return_type(self):
        return Probability

    @property
    def numeric_type(self):
        return float

    def shape(self, device, shots):
        num_shot_elements = sum((s.copies for s in shots.shot_vector)) if shots.has_partitioned_shots else 1
        len_wires = len(self.wires)
        if len_wires == 0:
            len_wires = len(device.wires) if device.wires else 0
        dim = self._get_num_basis_states(len_wires, device)
        return (dim,) if num_shot_elements == 1 else tuple(((dim,) for _ in range(num_shot_elements)))

    def process_samples(self, samples: Sequence[complex], wire_order: Wires, shot_range: Tuple[int]=None, bin_size: int=None):
        wire_map = dict(zip(wire_order, range(len(wire_order))))
        mapped_wires = [wire_map[w] for w in self.wires]
        if shot_range is not None:
            samples = samples[..., slice(*shot_range), :]
        if mapped_wires:
            samples = samples[..., mapped_wires]
        num_wires = qml.math.shape(samples)[-1]
        powers_of_two = 2 ** qml.math.arange(num_wires)[::-1]
        indices = samples @ powers_of_two
        batch_size = samples.shape[0] if qml.math.ndim(samples) == 3 else None
        dim = 2 ** num_wires
        new_bin_size = bin_size or samples.shape[-2]
        new_shape = (-1, new_bin_size) if batch_size is None else (batch_size, -1, new_bin_size)
        indices = indices.reshape(new_shape)
        prob = self._count_samples(indices, batch_size, dim)
        return qml.math.squeeze(prob) if bin_size is None else prob

    def process_state(self, state: Sequence[complex], wire_order: Wires):
        prob = qml.math.real(state) ** 2 + qml.math.imag(state) ** 2
        if self.wires == Wires([]):
            return prob
        inactive_wires = Wires.unique_wires([wire_order, self.wires])
        wire_map = dict(zip(wire_order, range(len(wire_order))))
        mapped_wires = [wire_map[w] for w in self.wires]
        inactive_wires = [wire_map[w] for w in inactive_wires]
        num_device_wires = len(wire_order)
        shape = [2] * num_device_wires
        desired_axes = np.argsort(np.argsort(mapped_wires))
        flat_shape = (-1,)
        expected_size = 2 ** num_device_wires
        batch_size = qml.math.get_batch_size(prob, (expected_size,), expected_size)
        if batch_size is not None:
            shape.insert(0, batch_size)
            inactive_wires = [idx + 1 for idx in inactive_wires]
            desired_axes = np.insert(desired_axes + 1, 0, 0)
            flat_shape = (batch_size, -1)
        prob = qml.math.reshape(prob, shape)
        prob = qml.math.sum(prob, axis=tuple(inactive_wires))
        prob = qml.math.transpose(prob, desired_axes)
        return qml.math.reshape(prob, flat_shape)

    def process_counts(self, counts: dict, wire_order: Wires) -> np.ndarray:
        wire_map = dict(zip(wire_order, range(len(wire_order))))
        mapped_wires = [wire_map[w] for w in self.wires]
        if mapped_wires:
            mapped_counts = {}
            for outcome, occurrence in counts.items():
                mapped_outcome = ''.join((outcome[i] for i in mapped_wires))
                mapped_counts[mapped_outcome] = mapped_counts.get(mapped_outcome, 0) + occurrence
            counts = mapped_counts
        num_shots = sum(counts.values())
        num_wires = len(next(iter(counts)))
        dim = 2 ** num_wires
        prob_vector = qml.math.zeros(dim, dtype='float64')
        for outcome, occurrence in counts.items():
            prob_vector[int(outcome, base=2)] = occurrence / num_shots
        return prob_vector

    @staticmethod
    def _count_samples(indices, batch_size, dim):
        """Count the occurrences of sampled indices and convert them to relative
        counts in order to estimate their occurrence probability."""
        num_bins, bin_size = indices.shape[-2:]
        if batch_size is None:
            prob = qml.math.zeros((dim, num_bins), dtype='float64')
            for b, idx in enumerate(indices):
                basis_states, counts = qml.math.unique(idx, return_counts=True)
                prob[basis_states, b] = counts / bin_size
            return prob
        prob = qml.math.zeros((batch_size, dim, num_bins), dtype='float64')
        indices = indices.reshape((batch_size, num_bins, bin_size))
        for i, _indices in enumerate(indices):
            for b, idx in enumerate(_indices):
                basis_states, counts = qml.math.unique(idx, return_counts=True)
                prob[i, basis_states, b] = counts / bin_size
        return prob