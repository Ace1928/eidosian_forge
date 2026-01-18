import numpy as np
import cirq
def test_relative_luminance():
    rl = cirq.vis.relative_luminance([100, 100, 100])
    assert np.isclose(rl, 55560.636)
    rl = cirq.vis.relative_luminance([0, 1, 2])
    assert np.isclose(rl, 1.0728676632649454)
    rl = cirq.vis.relative_luminance(np.array([0, 1, 2]))
    assert np.isclose(rl, 1.0728676632649454)