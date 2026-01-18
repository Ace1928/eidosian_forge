from typing import Dict, List, Optional, Set, TYPE_CHECKING
import abc
import collections
from cirq.devices import GridQubit
from cirq_google.line.placement import place_strategy
from cirq_google.line.placement.chip import chip_as_adjacency_list
from cirq_google.line.placement.sequence import GridQubitLineTuple
Runs line sequence search.

        Args:
            device: Chip description.
            length: Required line length.

        Returns:
            Linear sequences found on the chip.

        Raises:
            ValueError: If search algorithm passed on initialization is not
                        recognized.
        