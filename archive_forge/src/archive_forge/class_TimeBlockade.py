from __future__ import annotations
from abc import ABC
from qiskit.pulse import channels as chans
from qiskit.pulse.instructions import instruction
from qiskit.pulse.exceptions import PulseError
class TimeBlockade(Directive):
    """Pulse ``TimeBlockade`` directive.

    This instruction is intended to be used internally within the pulse builder,
    to convert :class:`.Schedule` into :class:`.ScheduleBlock`.
    Because :class:`.ScheduleBlock` cannot take an absolute instruction time interval,
    this directive helps the block representation to find the starting time of an instruction.

    Example:

        This schedule plays constant pulse at t0 = 120.

        .. code-block:: python

            schedule = Schedule()
            schedule.insert(120, Play(Constant(10, 0.1), DriveChannel(0)))

        This schedule block is expected to be identical to above at a time of execution.

        .. code-block:: python

            block = ScheduleBlock()
            block.append(TimeBlockade(120, DriveChannel(0)))
            block.append(Play(Constant(10, 0.1), DriveChannel(0)))

        Such conversion may be done by

        .. code-block:: python

            from qiskit.pulse.transforms import block_to_schedule, remove_directives

            schedule = remove_directives(block_to_schedule(block))


    .. note::

        The TimeBlockade instruction behaves almost identically
        to :class:`~qiskit.pulse.instructions.Delay` instruction.
        However, the TimeBlockade is just a compiler directive and must be removed before execution.
        This may be done by :func:`~qiskit.pulse.transforms.remove_directives` transform.
        Once these directives are removed, occupied timeslots are released and
        user can insert another instruction without timing overlap.
    """

    def __init__(self, duration: int, channel: chans.Channel, name: str | None=None):
        """Create a time blockade directive.

        Args:
            duration: Length of time of the occupation in terms of dt.
            channel: The channel that will be the occupied.
            name: Name of the time blockade for display purposes.
        """
        super().__init__(operands=(duration, channel), name=name)

    def _validate(self):
        """Called after initialization to validate instruction data.

        Raises:
            PulseError: If the input ``duration`` is not integer value.
        """
        if not isinstance(self.duration, int):
            raise PulseError('TimeBlockade duration cannot be parameterized. Specify an integer duration value.')

    @property
    def channel(self) -> chans.Channel:
        """Return the :py:class:`~qiskit.pulse.channels.Channel` that this instruction is
        scheduled on.
        """
        return self.operands[1]

    @property
    def channels(self) -> tuple[chans.Channel]:
        """Returns the channels that this schedule uses."""
        return (self.channel,)

    @property
    def duration(self) -> int:
        """Duration of this instruction."""
        return self.operands[0]