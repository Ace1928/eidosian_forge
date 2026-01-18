import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
@pytest.mark.parametrize('label_map', label_maps)
def test_wire_label_in_tensor_prod_observables(self, device, label_map, tol, skip_if):
    """Test that when given a tensor observable the variance is the same regardless of how the
        wires are labelled, as long as they match the device order.

        eg:
        dev1 = qml.device("default.qubit", wires=[0, 1, 2])
        dev2 = qml.device("default.qubit", wires=['c', 'b', 'a']

        def circ(wire_labels):
            return qml.var(qml.Z(wire_labels[0]) @ qml.X(wire_labels[2]))

        c1, c2 = qml.QNode(circ, dev1), qml.QNode(circ, dev2)
        c1([0, 1, 2]) == c2(['c', 'b', 'a'])
        """
    dev = device(wires=3)
    dev_custom_labels = device(wires=label_map)
    if isinstance(dev, qml.Device):
        skip_if(dev, {'supports_tensor_observables': False})

    def circ(wire_labels):
        sub_routine(wire_labels)
        return qml.var(qml.X(wire_labels[0]) @ qml.Y(wire_labels[1]) @ qml.Z(wire_labels[2]))
    circ_base_label = qml.QNode(circ, device=dev)
    circ_custom_label = qml.QNode(circ, device=dev_custom_labels)
    assert np.allclose(circ_base_label(wire_labels=range(3)), circ_custom_label(wire_labels=label_map), atol=tol(dev.shots), rtol=0)