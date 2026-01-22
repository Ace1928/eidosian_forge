from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable, List, Sequence, Mapping, Optional, Set, Tuple, cast
from warnings import warn
from pyquil.quil import Program
from pyquil.quilatom import ParameterDesignator, QubitDesignator, format_parameter
from pyquil.quilbase import (
def _build_custom_source_target_op(self) -> None:
    """
        Update the partial diagram with a single operation involving a source and a target
        (e.g. a controlled gate, a swap).

        Advances the index by one.
        """
    assert self.working_instructions is not None
    instr = self.working_instructions[self.index]
    assert isinstance(instr, Gate)
    source, target = qubit_indices(instr)
    assert self.diagram is not None
    displaced = self.diagram.interval(min(source, target), max(source, target))
    self.diagram.extend_lines_to_common_edge(displaced)
    source_op, target_op = SOURCE_TARGET_OP[instr.name]
    offset = (-1 if source > target else 1) * (len(displaced) - 1)
    self.diagram.append(source, source_op(source, offset))
    self.diagram.append(target, target_op())
    self.diagram.extend_lines_to_common_edge(displaced)
    self.index += 1