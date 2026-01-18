from functools import reduce
from typing import List, NamedTuple, Sequence, Tuple
from dataclasses import dataclass
import numpy as np
import cirq
from cirq import value
from cirq._compat import proper_repr, proper_eq
from cirq.transformers.heuristic_decompositions.gate_tabulation_math_utils import (
def two_qubit_gate_product_tabulation(base_gate: np.ndarray, max_infidelity: float, *, sample_scaling: int=50, allow_missed_points: bool=True, random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> TwoQubitGateTabulation:
    """Generate a TwoQubitGateTabulation for a base two qubit unitary.

    Args:
        base_gate: The base gate of the tabulation.
        max_infidelity: Sets the desired density of tabulated product unitaries.
            The typical nearest neighbor Euclidean spacing (of the KAK vectors)
            will be on the order of $\\sqrt{max\\_infidelity}$. Thus the number of
            tabulated points will scale as $max\\_infidelity^{-3/2}$.
        sample_scaling: Relative number of random gate products to use in the
            tabulation. The total number of random local unitaries scales as
            ~ $max\\_infidelity^{-3/2} * sample\\_scaling$. Must be positive.
        random_state: Random state or random state seed.
        allow_missed_points: If True, the tabulation is allowed to conclude
            even if not all points in the Weyl chamber are expected to be
            compilable using 2 or 3 base gates. Otherwise, an error is raised
            in this case.

    Returns:
        A TwoQubitGateTabulation object used to compile new two-qubit gates from
        products of the base gate with 1-local unitaries.

    Raises:
        ValueError: If `allow_missing_points` is False and not all the points
            in the Weyl chamber are compilable using 2 or 3 base gates.
    """
    rng = value.parse_random_state(random_state)
    assert 1 / 2 > max_infidelity > 0
    spacing = np.sqrt(max_infidelity / 3)
    mesh_points = weyl_chamber_mesh(spacing)
    assert sample_scaling > 0, 'Input sample_scaling must positive.'
    num_mesh_points = mesh_points.shape[0]
    num_samples = num_mesh_points * sample_scaling
    kak_vecs = [cirq.kak_vector(base_gate, check_preconditions=False)]
    sq_cycles: List[Tuple[_SingleQubitGatePair, ...]] = [()]
    u_locals_0 = random_qubit_unitary((num_samples,), rng=rng)
    u_locals_1 = random_qubit_unitary((num_samples,), rng=rng)
    tabulated_kak_inds = np.zeros((num_mesh_points,), dtype=bool)
    tabulation_cutoff = 0.5 * spacing
    out = _tabulate_kak_vectors(already_tabulated=tabulated_kak_inds, base_gate=base_gate, max_dist=tabulation_cutoff, kak_mesh=mesh_points, local_unitary_pairs=[(u_locals_0, u_locals_1)])
    kak_vecs.extend(out.kept_kaks)
    sq_cycles.extend(out.kept_cycles)
    kak_vecs_single = np.array(kak_vecs)
    sq_cycles_single = list(sq_cycles)
    summary = f'Fraction of Weyl chamber reached with 2 gates: {tabulated_kak_inds.sum() / num_mesh_points:.3f}'
    out = _tabulate_kak_vectors(already_tabulated=tabulated_kak_inds, base_gate=base_gate, max_dist=tabulation_cutoff, kak_mesh=mesh_points, local_unitary_pairs=[(u_locals_0, u_locals_1)] * 2)
    kak_vecs.extend(out.kept_kaks)
    sq_cycles.extend(out.kept_cycles)
    summary += f'\nFraction of Weyl chamber reached with 2 gates and 3 gates(same single qubit): {tabulated_kak_inds.sum() / num_mesh_points:.3f}'
    missing_vec_inds = np.logical_not(tabulated_kak_inds).nonzero()[0]
    if not np.any(missing_vec_inds):
        return TwoQubitGateTabulation(base_gate, np.array(kak_vecs), sq_cycles, max_infidelity, summary, ())
    u_locals_0p = random_qubit_unitary((100,), rng=rng)
    u_locals_1p = random_qubit_unitary((100,), rng=rng)
    u_locals = vector_kron(u_locals_0p, u_locals_1p)
    missed_points = []
    base_gate_dag = base_gate.conj().T
    for ind in missing_vec_inds:
        missing_vec = mesh_points[ind]
        missing_unitary = kak_vector_to_unitary(missing_vec)
        products = np.einsum('ab,...bc,cd', base_gate_dag, u_locals, missing_unitary)
        kaks = cirq.kak_vector(products, check_preconditions=False)
        kaks = kaks[..., np.newaxis, :]
        dists2 = np.sum((kaks - kak_vecs_single) ** 2, axis=-1)
        min_dist_inds = np.unravel_index(dists2.argmin(), dists2.shape)
        min_dist = np.sqrt(dists2[min_dist_inds])
        if min_dist < tabulation_cutoff:
            new_ind, old_ind = min_dist_inds
            new_product = products[new_ind]
            if old_ind == 0:
                assert not sq_cycles_single[old_ind]
                base_product = base_gate
                _, kL, actual = _outer_locals_for_unitary(new_product, base_product)
                sq_cycles.append((kL,))
            else:
                assert len(sq_cycles_single[old_ind]) == 1
                old_sq_cycle = sq_cycles_single[old_ind][0]
                old_k = np.kron(*old_sq_cycle)
                base_product = base_gate @ old_k @ base_gate
                _, kL, actual = _outer_locals_for_unitary(new_product, base_product)
                sq_cycles.append((old_sq_cycle, kL))
            kak_vecs.append(cirq.kak_vector(base_gate @ actual, check_preconditions=False))
        elif not allow_missed_points:
            raise ValueError(f'Failed to tabulate a KAK vector near {missing_vec}')
        else:
            missed_points.append(missing_vec)
    kak_vecs_arr = np.array(kak_vecs)
    summary += f'\nFraction of Weyl chamber reached with 2 gates and 3 gates (after patchup): {(len(kak_vecs_arr) - 1) / num_mesh_points:.3f}'
    return TwoQubitGateTabulation(base_gate, kak_vecs_arr, sq_cycles, max_infidelity, summary, tuple(missed_points))