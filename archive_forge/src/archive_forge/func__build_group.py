from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable, List, Sequence, Mapping, Optional, Set, Tuple, cast
from warnings import warn
from pyquil.quil import Program
from pyquil.quilatom import ParameterDesignator, QubitDesignator, format_parameter
from pyquil.quilbase import (
def _build_group(self) -> None:
    """
        Update the partial diagram with the subcircuit delimited by the grouping PRAGMA.

        Advances the index beyond the ending pragma.
        """
    assert self.working_instructions is not None
    instr = self.working_instructions[self.index]
    assert isinstance(instr, Pragma)
    if len(instr.args) != 0:
        raise ValueError(f'PRAGMA {PRAGMA_BEGIN_GROUP} expected a freeform string, or nothing at all.')
    start = self.index + 1
    for j in range(start, len(self.working_instructions)):
        instruction_j = self.working_instructions[j]
        if isinstance(instruction_j, Pragma) and instruction_j.command == PRAGMA_END_GROUP:
            block_settings = replace(self.settings, label_qubit_lines=False, qubit_line_open_wire_length=0)
            subcircuit = Program(*self.working_instructions[start:j])
            block = DiagramBuilder(subcircuit, block_settings).build()
            block_name = instr.freeform_string if instr.freeform_string else ''
            assert self.diagram is not None
            self.diagram.append_diagram(block, group=block_name)
            self.index = j + 1
            return
    raise ValueError('Unable to find PRAGMA {} matching {}.'.format(PRAGMA_END_GROUP, instr))