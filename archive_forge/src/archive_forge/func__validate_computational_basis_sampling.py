import copy
from threading import RLock
import pennylane as qml
from pennylane.measurements import CountsMP, ProbabilityMP, SampleMP, MeasurementProcess
from pennylane.operation import DecompositionUndefinedError, Operator, StatePrepBase
from pennylane.queuing import AnnotatedQueue, QueuingManager, process_queue
from pennylane.pytrees import register_pytree
from .qscript import QuantumScript
def _validate_computational_basis_sampling(measurements):
    """Auxiliary function for validating computational basis state sampling with other measurements considering the
    qubit-wise commutativity relation."""
    non_comp_basis_sampling_obs = []
    comp_basis_sampling_obs = []
    for o in measurements:
        if o.samples_computational_basis:
            comp_basis_sampling_obs.append(o)
        else:
            non_comp_basis_sampling_obs.append(o)
    if non_comp_basis_sampling_obs:
        all_wires = []
        empty_wires = qml.wires.Wires([])
        for idx, cb_obs in enumerate(comp_basis_sampling_obs):
            if cb_obs.wires == empty_wires:
                all_wires = qml.wires.Wires.all_wires([m.wires for m in measurements])
                break
            all_wires.append(cb_obs.wires)
            if idx == len(comp_basis_sampling_obs) - 1:
                all_wires = qml.wires.Wires.all_wires(all_wires)
        with QueuingManager.stop_recording():
            pauliz_for_cb_obs = qml.Z(all_wires) if len(all_wires) == 1 else qml.operation.Tensor(*[qml.Z(w) for w in all_wires])
        for obs in non_comp_basis_sampling_obs:
            if obs.obs is not None and (not qml.pauli.utils.are_pauli_words_qwc([obs.obs, pauliz_for_cb_obs])):
                raise qml.QuantumFunctionError(_err_msg_for_some_meas_not_qwc(measurements))