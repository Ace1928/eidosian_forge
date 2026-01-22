import abc
import enum
import html
import itertools
import math
from collections import defaultdict
from typing import (
from typing_extensions import Self
import networkx
import numpy as np
import cirq._version
from cirq import _compat, devices, ops, protocols, qis
from cirq._doc import document
from cirq.circuits._bucket_priority_queue import BucketPriorityQueue
from cirq.circuits.circuit_operation import CircuitOperation
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.circuits.qasm_output import QasmOutput
from cirq.circuits.text_diagram_drawer import TextDiagramDrawer
from cirq.circuits.moment import Moment
from cirq.protocols import circuit_diagram_info_protocol
from cirq.type_workarounds import NotImplementedType
def _pick_or_create_inserted_op_moment_index(self, splitter_index: int, op: 'cirq.Operation', strategy: 'cirq.InsertStrategy') -> int:
    """Determines and prepares where an insertion will occur.

        Args:
            splitter_index: The index to insert at.
            op: The operation that will be inserted.
            strategy: The insertion strategy.

        Returns:
            The index of the (possibly new) moment where the insertion should
                occur.

        Raises:
            ValueError: Unrecognized append strategy.
        """
    if strategy is InsertStrategy.NEW or strategy is InsertStrategy.NEW_THEN_INLINE:
        self._moments.insert(splitter_index, Moment())
        self._mutated()
        return splitter_index
    if strategy is InsertStrategy.INLINE:
        if 0 <= splitter_index - 1 < len(self._moments) and self._can_add_op_at(splitter_index - 1, op):
            return splitter_index - 1
        return self._pick_or_create_inserted_op_moment_index(splitter_index, op, InsertStrategy.NEW)
    if strategy is InsertStrategy.EARLIEST:
        if self._can_add_op_at(splitter_index, op):
            return self.earliest_available_moment(op, end_moment_index=splitter_index)
        return self._pick_or_create_inserted_op_moment_index(splitter_index, op, InsertStrategy.INLINE)
    raise ValueError(f'Unrecognized append strategy: {strategy}')