from dataclasses import dataclass
from typing import Callable, List, Union
import numpy as np
import pennylane as qml
from pennylane.wires import Wires
from pennylane.operation import Operator
from pennylane.ops.qubit.hamiltonian import Hamiltonian
from .parametrized_hamiltonian import ParametrizedHamiltonian
class HardwareHamiltonian(ParametrizedHamiltonian):
    """Internal class used to keep track of the required information to translate a ``ParametrizedHamiltonian``
    into hardware.

    This class contains the ``coeffs`` and the ``observables`` to construct the :class:`ParametrizedHamiltonian`,
    but on top of that also contains attributes that store parameteres relevant for real hardware execution.

    .. warning::

        This class should NEVER be initialized directly! Please use the functions
        :func:`rydberg_interaction` and :func:`drive` instead.

    .. seealso:: :func:`rydberg_interaction`, :func:`drive`, :class:`ParametrizedHamiltonian`

    Args:
        coeffs (Union[float, callable]): coefficients of the Hamiltonian expression, which may be
            constants or parametrized functions. All functions passed as ``coeffs`` must have two
            arguments, the first one being the trainable parameters and the second one being time.
        observables (Iterable[Observable]): observables in the Hamiltonian expression, of same
            length as ``coeffs``

    Keyword Args:
        reorder_fn (callable): function for reordering the parameters before calling.
            This allows automatically copying parameters when they are used for different terms,
            as well as allowing single terms to depend on multiple parameters, as is the case for
            drive Hamiltonians. Note that in order to add two HardwareHamiltonians,
            the reorder_fn needs to be matching.
        settings Union[RydbergSettings, TransmonSettings]: Dataclass containing the hardware specific settings. Default is ``None``.
        pulses (list[HardwarePulse]): list of ``HardwarePulse`` dataclasses containing the information about the
            amplitude, phase, drive frequency and wires of each pulse

    Returns:
        HardwareHamiltonian: class representing the Hamiltonian of Rydberg or Transmon device.

    """

    def __init__(self, coeffs, observables, reorder_fn: Callable=_reorder_parameters, pulses: List['HardwarePulse']=None, settings: Union['RydbergSettings', 'TransmonSettings']=None):
        self.settings = settings
        self.pulses = [] if pulses is None else pulses
        self.reorder_fn = reorder_fn
        super().__init__(coeffs, observables)

    def __call__(self, params, t):
        params = self.reorder_fn(params, self.coeffs_parametrized)
        return super().__call__(params, t)

    def __repr__(self):
        return f'HardwareHamiltonian: terms={qml.math.shape(self.coeffs)[0]}'

    def __add__(self, other):
        if isinstance(other, HardwareHamiltonian):
            if not self.reorder_fn == other.reorder_fn:
                raise ValueError(f'Cannot add two HardwareHamiltonians with different reorder functions. Received reorder_fns {self.reorder_fn} and {other.reorder_fn}. This is likely due to an attempt to add hardware compatible Hamiltonians for different target systems.')
            if self.settings is None and other.settings is None:
                new_settings = None
            else:
                new_settings = self.settings + other.settings
            new_pulses = self.pulses + other.pulses
            new_ops = self.ops + other.ops
            new_coeffs = self.coeffs + other.coeffs
            return HardwareHamiltonian(new_coeffs, new_ops, reorder_fn=self.reorder_fn, settings=new_settings, pulses=new_pulses)
        ops = self.ops.copy()
        coeffs = self.coeffs.copy()
        settings = self.settings
        pulses = self.pulses
        if isinstance(other, (Hamiltonian, ParametrizedHamiltonian)):
            new_coeffs = coeffs + list(other.coeffs.copy())
            new_ops = ops + other.ops.copy()
            return HardwareHamiltonian(new_coeffs, new_ops, reorder_fn=self.reorder_fn, settings=settings, pulses=pulses)
        if isinstance(other, qml.ops.SProd):
            new_coeffs = coeffs + [other.scalar]
            new_ops = ops + [other.base]
            return HardwareHamiltonian(new_coeffs, new_ops, reorder_fn=self.reorder_fn, settings=settings, pulses=pulses)
        if isinstance(other, Operator):
            new_coeffs = coeffs + [1]
            new_ops = ops + [other]
            return HardwareHamiltonian(new_coeffs, new_ops, reorder_fn=self.reorder_fn, settings=settings, pulses=pulses)
        if isinstance(other, (int, float)):
            if other in (0, 0.0):
                return HardwareHamiltonian(coeffs, ops, reorder_fn=self.reorder_fn, settings=settings, pulses=pulses)
            new_coeffs = coeffs + [other]
            with qml.queuing.QueuingManager.stop_recording():
                new_ops = ops + [qml.Identity(self.wires[0])]
            return HardwareHamiltonian(new_coeffs, new_ops, reorder_fn=self.reorder_fn, settings=settings, pulses=pulses)
        return NotImplemented

    def __radd__(self, other):
        """Deals with the special case where a HardwareHamiltonian is added to a
        ParametrizedHamiltonian. Ensures that this returns a HardwareHamiltonian where
        the order of the parametrized coefficients and operators matches the order of
        the hamiltonians, i.e. that
        ParametrizedHamiltonian + HardwareHamiltonian
        returns a HardwareHamiltonian where the call expects params = [params_PH] + [params_RH]
        """
        if isinstance(other, ParametrizedHamiltonian):
            ops = self.ops.copy()
            coeffs = self.coeffs.copy()
            new_coeffs = other.coeffs.copy() + coeffs
            new_ops = other.ops.copy() + ops
            return HardwareHamiltonian(new_coeffs, new_ops, reorder_fn=self.reorder_fn, settings=self.settings, pulses=self.pulses)
        return self.__add__(other)