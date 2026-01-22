import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from typing import (
import bytecode as _bytecode
from bytecode.concrete import ConcreteInstr
from bytecode.flags import CompilerFlags
from bytecode.instr import UNSET, Instr, Label, SetLineno, TryBegin, TryEnd
def _compute_exception_handler_stack_usage(self, block: BasicBlock, push_lasti: bool) -> Generator[Union['_StackSizeComputer', int], int, None]:
    b_id = id(block)
    if self.minsize < self.common.exception_block_startsize[b_id]:
        block_size = (yield _StackSizeComputer(self.common, block, self.minsize, self.maxsize, self.minsize, push_lasti, None))
        self.common.exception_block_startsize[b_id] = self.minsize
        self.common.exception_block_maxsize[b_id] = block_size