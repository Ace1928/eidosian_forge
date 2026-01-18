import numpy as np
import cirq
import pytest
from cirq.experiments.single_qubit_readout_calibration_test import NoisySingleQubitReadoutSampler
def test_readout_confusion_matrix_raises():
    num_qubits = 2
    confusion_matrix = get_expected_cm(num_qubits, 0.1, 0.2)
    qubits = cirq.LineQubit.range(4)
    with pytest.raises(ValueError, match='measure_qubits cannot be empty'):
        _ = cirq.TensoredConfusionMatrices([], [], repetitions=0, timestamp=0)
    with pytest.raises(ValueError, match='len\\(confusion_matrices\\)'):
        _ = cirq.TensoredConfusionMatrices([confusion_matrix], [qubits[:2], qubits[2:]], repetitions=0, timestamp=0)
    with pytest.raises(ValueError, match='Shape mismatch for confusion matrix'):
        _ = cirq.TensoredConfusionMatrices(confusion_matrix, qubits, repetitions=0, timestamp=0)
    with pytest.raises(ValueError, match='Repeated qubits not allowed'):
        _ = cirq.TensoredConfusionMatrices([confusion_matrix, confusion_matrix], [qubits[:2], qubits[1:3]], repetitions=0, timestamp=0)
    readout_cm = cirq.TensoredConfusionMatrices([confusion_matrix, confusion_matrix], [qubits[:2], qubits[2:]], repetitions=0, timestamp=0)
    with pytest.raises(ValueError, match='should be a subset of'):
        _ = readout_cm.confusion_matrix([cirq.NamedQubit('a')])
    with pytest.raises(ValueError, match='should be a subset of'):
        _ = readout_cm.correction_matrix([cirq.NamedQubit('a')])
    with pytest.raises(ValueError, match='result.shape .* should be'):
        _ = readout_cm.apply(np.asarray([100]), qubits[:2])
    with pytest.raises(ValueError, match='method.* should be'):
        _ = readout_cm.apply(np.asarray([1 / 16] * 16), method='l1norm')