import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from typing import (
import bytecode as _bytecode
from bytecode.concrete import ConcreteInstr
from bytecode.flags import CompilerFlags
from bytecode.instr import UNSET, Instr, Label, SetLineno, TryBegin, TryEnd
class _StackSizeComputer:
    """Helper computing the stack usage for a single block."""
    common: _StackSizeComputationStorage
    block: BasicBlock
    size: int
    maxsize: int
    minsize: int
    exception_handler: Optional[bool]
    pending_try_begin: Optional[TryBegin]

    def __init__(self, common: _StackSizeComputationStorage, block: BasicBlock, size: int, maxsize: int, minsize: int, exception_handler: Optional[bool], pending_try_begin: Optional[TryBegin]) -> None:
        self.common = common
        self.block = block
        self.size = size
        self.maxsize = maxsize
        self.minsize = minsize
        self.exception_handler = exception_handler
        self.pending_try_begin = pending_try_begin
        self._current_try_begin = pending_try_begin

    def run(self) -> Generator[Union['_StackSizeComputer', int], int, None]:
        """Iterate over the block instructions to compute stack usage."""
        block_id = id(self.block)
        fingerprint = (self.size, self.exception_handler)
        if id(self.block) in self.common.seen_blocks or not self._is_stacksize_computation_relevant(block_id, fingerprint):
            yield self.maxsize
        self.common.seen_blocks.add(block_id)
        self.common.blocks_startsizes[block_id].add(fingerprint)
        if self.exception_handler is not None:
            self._update_size(0, 1 + self.exception_handler)
        for i, instr in enumerate(self.block):
            if isinstance(instr, SetLineno):
                continue
            if isinstance(instr, TryBegin):
                assert self._current_try_begin is None
                self.common.try_begins.append(instr)
                self._current_try_begin = instr
                self.minsize = self.size
                continue
            elif isinstance(instr, TryEnd):
                if instr.entry is not self._current_try_begin:
                    continue
                assert isinstance(instr.entry.target, BasicBlock)
                yield from self._compute_exception_handler_stack_usage(instr.entry.target, instr.entry.push_lasti)
                self._current_try_begin = None
                continue
            if instr.has_jump():
                effect = instr.pre_and_post_stack_effect(jump=True) if self.common.check_pre_and_post else (instr.stack_effect(jump=True), 0)
                taken_size, maxsize, minsize = _update_size(*effect, self.size, self.maxsize, self.minsize)
                assert isinstance(instr.arg, BasicBlock)
                maxsize = (yield _StackSizeComputer(self.common, instr.arg, taken_size, maxsize, minsize, None, None if instr.is_final() and self.block.get_trailing_try_end(i) else self._current_try_begin))
                self.maxsize = max(self.maxsize, maxsize)
                if instr.is_uncond_jump():
                    if (te := self.block.get_trailing_try_end(i)):
                        assert te.entry is self._current_try_begin
                        assert isinstance(te.entry.target, BasicBlock)
                        yield from self._compute_exception_handler_stack_usage(te.entry.target, te.entry.push_lasti)
                    self.common.seen_blocks.remove(id(self.block))
                    yield self.maxsize
            effect = instr.pre_and_post_stack_effect(jump=False) if self.common.check_pre_and_post else (instr.stack_effect(jump=False), 0)
            self._update_size(*effect)
            if instr.is_final():
                if (te := self.block.get_trailing_try_end(i)):
                    assert isinstance(te.entry.target, BasicBlock)
                    yield from self._compute_exception_handler_stack_usage(te.entry.target, te.entry.push_lasti)
                self.common.seen_blocks.remove(id(self.block))
                yield self.maxsize
        if self.block.next_block:
            self.maxsize = (yield _StackSizeComputer(self.common, self.block.next_block, self.size, self.maxsize, self.minsize, None, self._current_try_begin))
        self.common.seen_blocks.remove(id(self.block))
        yield self.maxsize
    _current_try_begin: Optional[TryBegin]

    def _update_size(self, pre_delta: int, post_delta: int) -> None:
        size, maxsize, minsize = _update_size(pre_delta, post_delta, self.size, self.maxsize, self.minsize)
        self.size = size
        self.minsize = minsize
        self.maxsize = maxsize

    def _compute_exception_handler_stack_usage(self, block: BasicBlock, push_lasti: bool) -> Generator[Union['_StackSizeComputer', int], int, None]:
        b_id = id(block)
        if self.minsize < self.common.exception_block_startsize[b_id]:
            block_size = (yield _StackSizeComputer(self.common, block, self.minsize, self.maxsize, self.minsize, push_lasti, None))
            self.common.exception_block_startsize[b_id] = self.minsize
            self.common.exception_block_maxsize[b_id] = block_size

    def _is_stacksize_computation_relevant(self, block_id: int, fingerprint: Tuple[int, Optional[bool]]) -> bool:
        if sys.version_info >= (3, 11):
            return fingerprint not in self.common.blocks_startsizes[block_id]
        elif (sizes := self.common.blocks_startsizes[block_id]):
            return fingerprint[0] > max((f[0] for f in sizes))
        else:
            return True