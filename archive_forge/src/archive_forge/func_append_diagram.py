from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable, List, Sequence, Mapping, Optional, Set, Tuple, cast
from warnings import warn
from pyquil.quil import Program
from pyquil.quilatom import ParameterDesignator, QubitDesignator, format_parameter
from pyquil.quilbase import (
def append_diagram(self, diagram: 'DiagramState', group: Optional[str]=None) -> 'DiagramState':
    """
        Add all operations represented by the given diagram to their
        corresponding qubit lines in this diagram.

        If group is not None, then a TIKZ_GATE_GROUP is created with the label indicated by group.
        """
    grouped_qubits = diagram.qubits
    diagram.extend_lines_to_common_edge(grouped_qubits)
    self.extend_lines_to_common_edge(grouped_qubits)
    corner_row = grouped_qubits[0]
    corner_col = len(self.lines[corner_row]) + 1
    group_width = diagram.width(corner_row) - 1
    for q in diagram.qubits:
        for op in diagram.lines[q]:
            self.append(q, op)
    if group is not None:
        self.lines[corner_row][corner_col] += ' ' + TIKZ_GATE_GROUP(grouped_qubits, group_width, group)
    return self