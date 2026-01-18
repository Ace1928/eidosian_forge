from typing import List, Sequence, Tuple
import numpy as np
import sympy
import cirq
from cirq.contrib.custom_simulators.custom_state_simulator import CustomStateSimulator
def test_product_state_mode():
    sim = CustomStateSimulator(ComputationalBasisSimProductState, split_untangled_states=True)
    circuit = create_test_circuit()
    r = sim.simulate(circuit)
    assert r.measurements == {'a': np.array([1]), 'b': np.array([2])}
    assert isinstance(r._final_simulator_state, cirq.SimulationProductState)
    assert len(r._final_simulator_state.sim_states) == 3
    assert r._final_simulator_state.create_merged_state()._state.basis == [2, 2]