import abc
import collections
from typing import (
import numpy as np
from cirq import circuits, ops, protocols, study, value, work
from cirq.sim.simulation_state_base import SimulationStateBase
class SimulatesFinalState(Generic[TSimulationTrialResult], metaclass=value.ABCMetaImplementAnyOneOf):
    """Simulator that allows access to the simulator's final state.

    Implementors of this interface should implement the simulate_sweep_iter
    method. This simulator only returns the state of the quantum system
    for the final step of a simulation. This simulator state may be a state
    vector, the density matrix, or another representation, depending on the
    implementation.  For simulators that also allow stepping through
    a circuit see `SimulatesIntermediateState`.
    """

    def simulate(self, program: 'cirq.AbstractCircuit', param_resolver: 'cirq.ParamResolverOrSimilarType'=None, qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT, initial_state: Any=None) -> TSimulationTrialResult:
        """Simulates the supplied Circuit.

        This method returns a result which allows access to the entire
        simulator's final state.

        Args:
            program: The circuit to simulate.
            param_resolver: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation. The form of
                this state depends on the simulation implementation. See
                documentation of the implementing class for details.

        Returns:
            SimulationTrialResults for the simulation. Includes the final state.
        """
        return self.simulate_sweep(program, study.ParamResolver(param_resolver), qubit_order, initial_state)[0]

    def simulate_sweep(self, program: 'cirq.AbstractCircuit', params: 'cirq.Sweepable', qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT, initial_state: Any=None) -> List[TSimulationTrialResult]:
        """Wraps computed states in a list.

        Prefer overriding `simulate_sweep_iter`.
        """
        return list(self.simulate_sweep_iter(program, params, qubit_order, initial_state))

    def _simulate_sweep_to_iter(self, program: 'cirq.AbstractCircuit', params: 'cirq.Sweepable', qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT, initial_state: Any=None) -> Iterator[TSimulationTrialResult]:
        if type(self).simulate_sweep == SimulatesFinalState.simulate_sweep:
            raise RecursionError('Must define either simulate_sweep or simulate_sweep_iter.')
        yield from self.simulate_sweep(program, params, qubit_order, initial_state)

    @value.alternative(requires='simulate_sweep', implementation=_simulate_sweep_to_iter)
    def simulate_sweep_iter(self, program: 'cirq.AbstractCircuit', params: 'cirq.Sweepable', qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT, initial_state: Any=None) -> Iterator[TSimulationTrialResult]:
        """Simulates the supplied Circuit.

        This method returns a result which allows access to the entire final
        simulator state. In contrast to simulate, this allows for sweeping
        over different parameter values.

        Args:
            program: The circuit to simulate.
            params: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation. The form of
                this state depends on the simulation implementation. See
                documentation of the implementing class for details.

        Returns:
            Iterator over SimulationTrialResults for this run, one for each
            possible parameter resolver.
        """
        raise NotImplementedError()