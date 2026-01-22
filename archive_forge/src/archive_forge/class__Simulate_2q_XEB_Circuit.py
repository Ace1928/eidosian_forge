from dataclasses import dataclass
from typing import List, Optional, Sequence, TYPE_CHECKING, Dict, Any
import numpy as np
import pandas as pd
from cirq import sim, value
class _Simulate_2q_XEB_Circuit:
    """Closure used in `simulate_2q_xeb_circuits` so it works with multiprocessing."""

    def __init__(self, simulator: 'cirq.SimulatesIntermediateState'):
        self.simulator = simulator

    def __call__(self, task: _Simulate2qXEBTask) -> List[Dict[str, Any]]:
        """Helper function for simulating a given (circuit, cycle_depth)."""
        circuit_i = task.circuit_i
        cycle_depths = set(task.cycle_depths)
        circuit = task.circuit
        param_resolver = task.param_resolver
        circuit_max_cycle_depth = (len(circuit) - 1) // 2
        if max(cycle_depths) > circuit_max_cycle_depth:
            raise ValueError('`circuit` was not long enough to compute all `cycle_depths`.')
        records: List[Dict[str, Any]] = []
        for moment_i, step_result in enumerate(self.simulator.simulate_moment_steps(circuit=circuit, param_resolver=param_resolver)):
            if moment_i % 2 == 1:
                continue
            cycle_depth = moment_i // 2
            if cycle_depth not in cycle_depths:
                continue
            psi = step_result.state_vector()
            pure_probs = value.state_vector_to_probabilities(psi)
            records += [{'circuit_i': circuit_i, 'cycle_depth': cycle_depth, 'pure_probs': pure_probs}]
        return records