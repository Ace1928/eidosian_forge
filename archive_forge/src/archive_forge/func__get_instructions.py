import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from typing import (
import bytecode as _bytecode
from bytecode.concrete import ConcreteInstr
from bytecode.flags import CompilerFlags
from bytecode.instr import UNSET, Instr, Label, SetLineno, TryBegin, TryEnd
def _get_instructions(self) -> List:
    instructions: List = []
    try_begins: Dict[TryBegin, int] = {}
    for block in self:
        for index, instr in enumerate(block):
            if isinstance(instr, TryBegin):
                assert isinstance(instr.target, BasicBlock)
                try_begins.setdefault(instr, len(try_begins))
                instructions.append(('TryBegin', try_begins[instr], self.get_block_index(instr.target), instr.push_lasti))
            elif isinstance(instr, TryEnd):
                instructions.append(('TryEnd', try_begins[instr.entry]))
            elif isinstance(instr, Instr) and (instr.has_jump() or instr.is_final()):
                if instr.has_jump():
                    target_block = instr.arg
                    assert isinstance(target_block, BasicBlock)
                    c_instr = ConcreteInstr(instr.name, self.get_block_index(target_block), location=instr.location)
                    instructions.append(c_instr)
                else:
                    instructions.append(instr)
                if (te := block.get_trailing_try_end(index)):
                    instructions.append(('TryEnd', try_begins[te.entry]))
                break
            else:
                instructions.append(instr)
    return instructions