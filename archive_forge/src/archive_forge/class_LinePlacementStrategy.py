from typing import TYPE_CHECKING
import abc
from cirq_google.line.placement.sequence import GridQubitLineTuple
class LinePlacementStrategy(metaclass=abc.ABCMeta):
    """Choice and options for the line placement calculation method.

    Currently two methods are available: cirq.line.GreedySequenceSearchMethod
    and cirq.line.AnnealSequenceSearchMethod.
    """

    @abc.abstractmethod
    def place_line(self, device: 'cirq_google.GridDevice', length: int) -> GridQubitLineTuple:
        """Runs line sequence search.

        Args:
            device: Chip description.
            length: Required line length.

        Returns:
            Linear sequences found on the chip.
        """