from typing import Sequence, Callable
from functools import partial
import numpy as np
import pennylane as qml
from pennylane.gradients.metric_tensor import _get_aux_wire
from pennylane import transform
from pennylane.gradients.gradient_transform import _contract_qjac_with_cjac
from pennylane.transforms.tape_expand import expand_invalid_trainable_hadamard_gradient
from .gradient_transform import (
def _expval_hadamard_grad(tape, argnum, aux_wire):
    """Compute the Hadamard test gradient of a tape that returns an expectation value (probabilities are expectations
    values) with respect to a given set of all trainable gate parameters.
    The auxiliary wire is the wire which is used to apply the Hadamard gates and controlled gates.
    """
    argnums = argnum or tape.trainable_params
    g_tapes = []
    coeffs = []
    gradient_data = []
    measurements_probs = [idx for idx, m in enumerate(tape.measurements) if isinstance(m, qml.measurements.ProbabilityMP)]
    for trainable_param_idx, _ in enumerate(tape.trainable_params):
        if trainable_param_idx not in argnums:
            gradient_data.append(0)
            continue
        trainable_op, idx, p_idx = tape.get_operation(trainable_param_idx)
        ops_to_trainable_op = tape.operations[:idx + 1]
        ops_after_trainable_op = tape.operations[idx + 1:]
        sub_coeffs, generators = _get_generators(trainable_op)
        coeffs.extend(sub_coeffs)
        num_tape = 0
        for gen in generators:
            if isinstance(trainable_op, qml.Rot):
                if p_idx == 0:
                    op_before_trainable_op = ops_to_trainable_op.pop(-1)
                    ops_after_trainable_op = [op_before_trainable_op] + ops_after_trainable_op
                elif p_idx == 1:
                    ops_to_add_before = [qml.RZ(-trainable_op.data[2], wires=trainable_op.wires), qml.RX(np.pi / 2, wires=trainable_op.wires)]
                    ops_to_trainable_op.extend(ops_to_add_before)
                    ops_to_add_after = [qml.RX(-np.pi / 2, wires=trainable_op.wires), qml.RZ(trainable_op.data[2], wires=trainable_op.wires)]
                    ops_after_trainable_op = ops_to_add_after + ops_after_trainable_op
            ctrl_gen = [qml.ctrl(gen, control=aux_wire)]
            hadamard = [qml.Hadamard(wires=aux_wire)]
            ops = ops_to_trainable_op + hadamard + ctrl_gen + hadamard + ops_after_trainable_op
            measurements = []
            for m in tape.measurements:
                if isinstance(m.obs, qml.operation.Tensor):
                    obs_new = m.obs.obs.copy()
                elif m.obs:
                    obs_new = [m.obs]
                else:
                    obs_new = [qml.Z(i) for i in m.wires]
                obs_new.append(qml.Y(aux_wire))
                obs_new = qml.operation.Tensor(*obs_new)
                if isinstance(m, qml.measurements.ExpectationMP):
                    measurements.append(qml.expval(op=obs_new))
                else:
                    measurements.append(qml.probs(op=obs_new))
            new_tape = qml.tape.QuantumScript(ops=ops, measurements=measurements, shots=tape.shots)
            _rotations, _measurements = qml.tape.tape.rotations_and_diagonal_measurements(new_tape)
            new_tape._ops = new_tape.operations + _rotations
            new_tape._measurements = _measurements
            new_tape._update()
            num_tape += 1
            g_tapes.append(new_tape)
        gradient_data.append(num_tape)
    multi_measurements = len(tape.measurements) > 1
    multi_params = len(tape.trainable_params) > 1

    def processing_fn(results):
        """Post processing function for computing a hadamard gradient."""
        final_res = []
        for coeff, res in zip(coeffs, results):
            if isinstance(res, tuple):
                new_val = [qml.math.convert_like(2 * coeff * r, r) for r in res]
            else:
                new_val = qml.math.convert_like(2 * coeff * res, res)
            final_res.append(new_val)
        if measurements_probs:
            projector = np.array([1, -1])
            like = final_res[0][0] if multi_measurements else final_res[0]
            projector = qml.math.convert_like(projector, like)
            for idx, res in enumerate(final_res):
                if multi_measurements:
                    for prob_idx in measurements_probs:
                        num_wires_probs = len(tape.measurements[prob_idx].wires)
                        res_reshaped = qml.math.reshape(res[prob_idx], (2 ** num_wires_probs, 2))
                        final_res[idx][prob_idx] = qml.math.tensordot(res_reshaped, projector, axes=[[1], [0]])
                else:
                    prob_idx = measurements_probs[0]
                    num_wires_probs = len(tape.measurements[prob_idx].wires)
                    res = qml.math.reshape(res, (2 ** num_wires_probs, 2))
                    final_res[idx] = qml.math.tensordot(res, projector, axes=[[1], [0]])
        grads = []
        idx = 0
        for num_tape in gradient_data:
            if num_tape == 0:
                grads.append(qml.math.zeros(()))
            elif num_tape == 1:
                grads.append(final_res[idx])
                idx += 1
            else:
                result = final_res[idx:idx + num_tape]
                if multi_measurements:
                    grads.append([qml.math.array(qml.math.sum(res, axis=0)) for res in zip(*result)])
                else:
                    grads.append(qml.math.array(qml.math.sum(result)))
                idx += num_tape
        if not multi_measurements and (not multi_params):
            return grads[0]
        if not (multi_params and multi_measurements):
            if multi_measurements:
                return tuple(grads[0])
            return tuple(grads)
        grads_reorder = [[0] * len(tape.trainable_params) for _ in range(len(tape.measurements))]
        for i in range(len(tape.measurements)):
            for j in range(len(tape.trainable_params)):
                grads_reorder[i][j] = grads[j][i]
        grads_tuple = tuple((tuple(elem) for elem in grads_reorder))
        return grads_tuple
    return (g_tapes, processing_fn)