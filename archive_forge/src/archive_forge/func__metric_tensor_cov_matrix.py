from typing import Sequence, Callable
import functools
from functools import partial
import warnings
import numpy as np
import pennylane as qml
from pennylane.circuit_graph import LayerData
from pennylane.queuing import WrappedObj
from pennylane.transforms import transform
def _metric_tensor_cov_matrix(tape, argnum, diag_approx):
    """This is the metric tensor method for the block diagonal, using
    the covariance matrix of the generators of each layer.

    Args:
        tape (pennylane.QNode or .QuantumTape): quantum tape or QNode to find the metric tensor of
        argnum (list[int]): Trainable tape-parameter indices with respect to which the metric tensor
            is computed.
        diag_approx (bool): if True, use the diagonal approximation. If ``False`` , a
            block-diagonal approximation of the metric tensor is computed.
    Returns:
        list[pennylane.tape.QuantumTape]: Transformed tapes that compute the probabilities
            required for the covariance matrix
        callable: Post-processing function that computes the covariance matrix from the
            results of the tapes in the first return value
        list[list[.Observable]]: Observables measured in each tape, one inner list
            corresponding to one tape in the first return value
        list[list[float]]: Coefficients to scale the results for each observable, one inner list
            corresponding to one tape in the first return value
        list[list[bool]]: Each inner list corresponds to one tape and therefore also one parametrized
            layer and its elements determine whether a trainable parameter in that layer is
            contained in ``argnum``.
        list[None, int]: Id list representing the layer for each parameter.
        list[None, int]: Id list representing the observables for each parameter.


    This method assumes the ``generator`` of all parametrized operations with respect to
    which the tensor is computed to be Hermitian. This is the case for unitary single-parameter
    operations.
    """
    graph = tape.graph
    metric_tensor_tapes = []
    obs_list = []
    coeffs_list = []
    params_list = []
    in_argnum_list = []
    layers_ids = []
    obs_ids = []
    i = 0
    for queue, curr_ops, param_idx, _ in graph.iterate_parametrized_layers():
        params_list.append(param_idx)
        in_argnum_list.append([p in argnum for p in param_idx])
        if not any(in_argnum_list[-1]):
            layers_ids.extend([None] * len(in_argnum_list[-1]))
            obs_ids.extend([None] * len(in_argnum_list[-1]))
            continue
        layer_coeffs, layer_obs = ([], [])
        j = 0
        for p, op in zip(param_idx, curr_ops):
            layers_ids.append(i)
            if p in argnum:
                obs, s = qml.generator(op)
                layer_obs.append(obs)
                layer_coeffs.append(s)
                obs_ids.append(j)
                j = j + 1
            else:
                obs_ids.append(None)
        i = i + 1
        coeffs_list.append(layer_coeffs)
        obs_list.append(layer_obs)
        with qml.queuing.AnnotatedQueue() as layer_q:
            for op in queue:
                qml.apply(op)
            for o, param_in_argnum in zip(layer_obs, in_argnum_list[-1]):
                if param_in_argnum:
                    o.diagonalizing_gates()
            qml.probs(wires=tape.wires)
        layer_tape = qml.tape.QuantumScript.from_queue(layer_q)
        metric_tensor_tapes.append(layer_tape)

    def processing_fn(probs):
        gs = []
        probs_idx = 0
        for params_in_argnum in in_argnum_list:
            if not any(params_in_argnum):
                dim = len(params_in_argnum)
                gs.append(qml.math.zeros((dim, dim)))
                continue
            coeffs = coeffs_list[probs_idx]
            obs = obs_list[probs_idx]
            p = probs[probs_idx]
            scale = qml.math.convert_like(qml.math.outer(coeffs, coeffs), p)
            scale = qml.math.cast_like(scale, p)
            g = scale * qml.math.cov_matrix(p, obs, wires=tape.wires, diag_approx=diag_approx)
            for i, in_argnum in enumerate(params_in_argnum):
                if not in_argnum:
                    dim = g.shape[0]
                    g = qml.math.concatenate((g[:i], qml.math.zeros((1, dim)), g[i:]))
                    g = qml.math.concatenate((g[:, :i], qml.math.zeros((dim + 1, 1)), g[:, i:]), axis=1)
            gs.append(g)
            probs_idx += 1
        return qml.math.block_diag(gs)
    return (metric_tensor_tapes, processing_fn, obs_list, coeffs_list, in_argnum_list, layers_ids, obs_ids)