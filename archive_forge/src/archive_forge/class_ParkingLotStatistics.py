from __future__ import annotations
import math
from collections import OrderedDict
from typing import TYPE_CHECKING
import attrs
from .. import _core
from .._util import final
@attrs.frozen
class ParkingLotStatistics:
    """An object containing debugging information for a ParkingLot.

    Currently, the following fields are defined:

    * ``tasks_waiting`` (int): The number of tasks blocked on this lot's
      :meth:`trio.lowlevel.ParkingLot.park` method.

    """
    tasks_waiting: int