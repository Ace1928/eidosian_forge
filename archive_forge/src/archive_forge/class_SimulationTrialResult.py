import abc
import collections
from typing import (
import numpy as np
from cirq import circuits, ops, protocols, study, value, work
from cirq.sim.simulation_state_base import SimulationStateBase
@value.value_equality(unhashable=True)
class SimulationTrialResult(Generic[TSimulatorState]):
    """Results of a simulation by a SimulatesFinalState.

    Unlike `cirq.Result`, a SimulationTrialResult contains the final
    simulator_state of the system. This simulator_state is dependent on the
    simulation implementation and may be, for example, the state vector
    or the density matrix of the system.

    Attributes:
        params: A ParamResolver of settings used for this result.
        measurements: A dictionary from measurement gate key to measurement
            results. Measurement results are a numpy ndarray of actual boolean
            measurement results (ordered by the qubits acted on by the
            measurement gate.)
    """

    def __init__(self, params: 'cirq.ParamResolver', measurements: Mapping[str, np.ndarray], final_simulator_state: TSimulatorState) -> None:
        """Initializes the `SimulationTrialResult` class.

        Args:
            params: A ParamResolver of settings used for this result.
            measurements: A mapping from measurement gate key to measurement
                results. Measurement results are a numpy ndarray of actual
                boolean measurement results (ordered by the qubits acted on by
                the measurement gate.)
            final_simulator_state: The final simulator state.
        """
        self._params = params
        self._measurements = measurements
        self._final_simulator_state = final_simulator_state

    @property
    def params(self) -> 'cirq.ParamResolver':
        return self._params

    @property
    def measurements(self) -> Mapping[str, np.ndarray]:
        return self._measurements

    def __repr__(self) -> str:
        return f'cirq.SimulationTrialResult(params={self.params!r}, measurements={self.measurements!r}, final_simulator_state={self._final_simulator_state!r})'

    def __str__(self) -> str:

        def bitstring(vals):
            separator = ' ' if np.max(vals) >= 10 else ''
            return separator.join((str(int(v)) for v in vals))
        results = sorted([(key, bitstring(val)) for key, val in self.measurements.items()])
        if not results:
            return '(no measurements)'
        return ' '.join([f'{key}={val}' for key, val in results])

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Text output in Jupyter."""
        if cycle:
            p.text('SimulationTrialResult(...)')
        else:
            p.text(str(self))

    def _value_equality_values_(self) -> Any:
        measurements = {k: v.tolist() for k, v in sorted(self.measurements.items())}
        return (self.params, measurements, self._final_simulator_state)

    @property
    def qubit_map(self) -> Mapping['cirq.Qid', int]:
        """A map from Qid to index used to define the ordering of the basis in
        the result.
        """
        return self._final_simulator_state.qubit_map

    def _qid_shape_(self) -> Tuple[int, ...]:
        return _qubit_map_to_shape(self.qubit_map)