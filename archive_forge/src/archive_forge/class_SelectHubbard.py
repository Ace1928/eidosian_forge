from typing import Collection, Optional, Sequence, Tuple, Union
from numpy.typing import NDArray
import attr
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import and_gate, apply_gate_to_lth_target, arithmetic_gates
from cirq_ft.algos import prepare_uniform_superposition as prep_u
from cirq_ft.algos import (
@attr.frozen
class SelectHubbard(select_and_prepare.SelectOracle):
    """The SELECT operation optimized for the 2D Hubbard model.

    In contrast to the arbitrary chemistry Hamiltonian, we:
     - explicitly consider the two dimensions of indices to permit optimization of the circuits.
     - dispense with the `theta` parameter.

    If neither $U$ nor $V$ is set we apply the kinetic terms of the Hamiltonian:

    $$
    -\\hop{X} \\quad p < q \\\\
    -\\hop{Y} \\quad p > q
    $$

    If $U$ is set we know $(p,\\alpha)=(q,\\beta)$ and apply the single-body term: $-Z_{p,\\alpha}$.
    If $V$ is set we know $p=q, \\alpha=0$, and $\\beta=1$ and apply the spin term:
    $Z_{p,\\alpha}Z_{p,\\beta}$

    The circuit for implementing $\\textit{C-SELECT}_{Hubbard}$ has a T-cost of $10 * N + log(N)$
    and $0$ rotations.


    Args:
        x_dim: the number of sites along the x axis.
        y_dim: the number of sites along the y axis.
        control_val: Optional bit specifying the control value for constructing a controlled
            version of this gate. Defaults to None, which means no control.

    Signature:
        control: A control bit for the entire gate.
        U: Whether we're applying the single-site part of the potential.
        V: Whether we're applying the pairwise part of the potential.
        p_x: First set of site indices, x component.
        p_y: First set of site indices, y component.
        alpha: First set of sites' spin indicator.
        q_x: Second set of site indices, x component.
        q_y: Second set of site indices, y component.
        beta: Second set of sites' spin indicator.
        target: The system register to apply the select operation.

    References:
        Section V. and Fig. 19 of https://arxiv.org/abs/1805.03662.
    """
    x_dim: int
    y_dim: int
    control_val: Optional[int] = None

    def __attrs_post_init__(self):
        if self.x_dim != self.y_dim:
            raise NotImplementedError('Currently only supports the case where x_dim=y_dim.')

    @cached_property
    def control_registers(self) -> Tuple[infra.Register, ...]:
        return () if self.control_val is None else (infra.Register('control', 1),)

    @cached_property
    def selection_registers(self) -> Tuple[infra.SelectionRegister, ...]:
        return (infra.SelectionRegister('U', 1, 2), infra.SelectionRegister('V', 1, 2), infra.SelectionRegister('p_x', (self.x_dim - 1).bit_length(), self.x_dim), infra.SelectionRegister('p_y', (self.y_dim - 1).bit_length(), self.y_dim), infra.SelectionRegister('alpha', 1, 2), infra.SelectionRegister('q_x', (self.x_dim - 1).bit_length(), self.x_dim), infra.SelectionRegister('q_y', (self.y_dim - 1).bit_length(), self.y_dim), infra.SelectionRegister('beta', 1, 2))

    @cached_property
    def target_registers(self) -> Tuple[infra.Register, ...]:
        return (infra.Register('target', self.x_dim * self.y_dim * 2),)

    @cached_property
    def signature(self) -> infra.Signature:
        return infra.Signature([*self.control_registers, *self.selection_registers, *self.target_registers])

    def decompose_from_registers(self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]) -> cirq.OP_TREE:
        p_x, p_y, q_x, q_y = (quregs['p_x'], quregs['p_y'], quregs['q_x'], quregs['q_y'])
        U, V, alpha, beta = (quregs['U'], quregs['V'], quregs['alpha'], quregs['beta'])
        control, target = (quregs.get('control', ()), quregs['target'])
        yield selected_majorana_fermion.SelectedMajoranaFermionGate(selection_regs=(infra.SelectionRegister('alpha', 1, 2), infra.SelectionRegister('p_y', self.signature.get_left('p_y').total_bits(), self.y_dim), infra.SelectionRegister('p_x', self.signature.get_left('p_x').total_bits(), self.x_dim)), control_regs=self.control_registers, target_gate=cirq.Y).on_registers(control=control, p_x=p_x, p_y=p_y, alpha=alpha, target=target)
        yield swap_network.MultiTargetCSwap.make_on(control=V, target_x=p_x, target_y=q_x)
        yield swap_network.MultiTargetCSwap.make_on(control=V, target_x=p_y, target_y=q_y)
        yield swap_network.MultiTargetCSwap.make_on(control=V, target_x=alpha, target_y=beta)
        q_selection_regs = (infra.SelectionRegister('beta', 1, 2), infra.SelectionRegister('q_y', self.signature.get_left('q_y').total_bits(), self.y_dim), infra.SelectionRegister('q_x', self.signature.get_left('q_x').total_bits(), self.x_dim))
        yield selected_majorana_fermion.SelectedMajoranaFermionGate(selection_regs=q_selection_regs, control_regs=self.control_registers, target_gate=cirq.X).on_registers(control=control, q_x=q_x, q_y=q_y, beta=beta, target=target)
        yield swap_network.MultiTargetCSwap.make_on(control=V, target_x=alpha, target_y=beta)
        yield swap_network.MultiTargetCSwap.make_on(control=V, target_x=p_y, target_y=q_y)
        yield swap_network.MultiTargetCSwap.make_on(control=V, target_x=p_x, target_y=q_x)
        yield (cirq.S(*control) ** (-1) if control else cirq.global_phase_operation(-1j))
        yield cirq.Z(*U).controlled_by(*control)
        target_qubits_for_apply_to_lth_gate = [target[np.ravel_multi_index((1, qy, qx), (2, self.y_dim, self.x_dim))] for qx in range(self.x_dim) for qy in range(self.y_dim)]
        yield apply_gate_to_lth_target.ApplyGateToLthQubit(selection_regs=(infra.SelectionRegister('q_y', self.signature.get_left('q_y').total_bits(), self.y_dim), infra.SelectionRegister('q_x', self.signature.get_left('q_x').total_bits(), self.x_dim)), nth_gate=lambda *_: cirq.Z, control_regs=infra.Register('control', 1 + infra.total_bits(self.control_registers))).on_registers(q_x=q_x, q_y=q_y, control=[*V, *control], target=target_qubits_for_apply_to_lth_gate)

    def controlled(self, num_controls: Optional[int]=None, control_values: Optional[Union[cirq.ops.AbstractControlValues, Sequence[Union[int, Collection[int]]]]]=None, control_qid_shape: Optional[Tuple[int, ...]]=None) -> 'SelectHubbard':
        if num_controls is None:
            num_controls = 1
        if control_values is None:
            control_values = [1] * num_controls
        if isinstance(control_values, Sequence) and isinstance(control_values[0], int) and (len(control_values) == 1) and (self.control_val is None):
            return SelectHubbard(self.x_dim, self.y_dim, control_val=control_values[0])
        raise NotImplementedError(f'Cannot create a controlled version of {self} with control_values={control_values}.')

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        info = super(SelectHubbard, self)._circuit_diagram_info_(args)
        if self.control_val is None:
            return info
        ctrl = ('@' if self.control_val else '@(0)',)
        return info.with_wire_symbols(ctrl + info.wire_symbols[0:1] + info.wire_symbols[2:])

    def __repr__(self) -> str:
        return f'cirq_ft.SelectHubbard({self.x_dim}, {self.y_dim}, {self.control_val})'