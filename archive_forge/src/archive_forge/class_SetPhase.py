from typing import Optional, Union, Tuple
from qiskit.circuit import ParameterExpression
from qiskit.pulse.channels import PulseChannel
from qiskit.pulse.instructions.instruction import Instruction
from qiskit.pulse.exceptions import PulseError
class SetPhase(Instruction):
    """The set phase instruction sets the phase of the proceeding pulses on that channel
    to ``phase`` radians.

    In particular, a PulseChannel creates pulses of the form

    .. math::

        Re[\\exp(i 2\\pi f jdt + \\phi) d_j]

    The ``SetPhase`` instruction sets :math:`\\phi` to the instruction's ``phase`` operand.
    """

    def __init__(self, phase: Union[complex, ParameterExpression], channel: PulseChannel, name: Optional[str]=None):
        """Instantiate a set phase instruction, setting the output signal phase on ``channel``
        to ``phase`` [radians].

        Args:
            phase: The rotation angle in radians.
            channel: The channel this instruction operates on.
            name: Display name for this instruction.
        """
        super().__init__(operands=(phase, channel), name=name)

    def _validate(self):
        """Called after initialization to validate instruction data.

        Raises:
            PulseError: If the input ``channel`` is not type :class:`PulseChannel`.
        """
        if not isinstance(self.channel, PulseChannel):
            raise PulseError(f'Expected a pulse channel, got {self.channel} instead.')

    @property
    def phase(self) -> Union[complex, ParameterExpression]:
        """Return the rotation angle enacted by this instruction in radians."""
        return self.operands[0]

    @property
    def channel(self) -> PulseChannel:
        """Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is
        scheduled on.
        """
        return self.operands[1]

    @property
    def channels(self) -> Tuple[PulseChannel]:
        """Returns the channels that this schedule uses."""
        return (self.channel,)

    @property
    def duration(self) -> int:
        """Duration of this instruction."""
        return 0