import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
class ControlFlowAnalysis(object):
    """
    Attributes
    ----------
    - bytecode

    - blocks

    - blockseq

    - doms: dict of set
        Dominators

    - backbone: set of block offsets
        The set of block that is common to all possible code path.

    """

    def __init__(self, bytecode):
        self.bytecode = bytecode
        self.blocks = {}
        self.liveblocks = {}
        self.blockseq = []
        self.doms = None
        self.backbone = None
        self._force_new_block = True
        self._curblock = None
        self._blockstack = []
        self._loops = []
        self._withs = []

    def iterblocks(self):
        """
        Return all blocks in sequence of occurrence
        """
        for i in self.blockseq:
            yield self.blocks[i]

    def iterliveblocks(self):
        """
        Return all live blocks in sequence of occurrence
        """
        for i in self.blockseq:
            if i in self.liveblocks:
                yield self.blocks[i]

    def incoming_blocks(self, block):
        """
        Yield (incoming block, number of stack pops) pairs for *block*.
        """
        for i, pops in block.incoming_jumps.items():
            if i in self.liveblocks:
                yield (self.blocks[i], pops)

    def dump(self, file=None):
        self.graph.dump(file=None)

    def run(self):
        for inst in self._iter_inst():
            fname = 'op_%s' % inst.opname
            fn = getattr(self, fname, None)
            if fn is not None:
                fn(inst)
            elif inst.is_jump:
                l = Loc(self.bytecode.func_id.filename, inst.lineno)
                if inst.opname in {'SETUP_FINALLY'}:
                    msg = "'try' block not supported until python3.7 or later"
                else:
                    msg = 'Use of unsupported opcode (%s) found' % inst.opname
                raise UnsupportedError(msg, loc=l)
            else:
                pass
        for cur, nxt in zip(self.blockseq, self.blockseq[1:]):
            blk = self.blocks[cur]
            if not blk.outgoing_jumps and (not blk.terminating):
                blk.outgoing_jumps[nxt] = 0
        graph = CFGraph()
        for b in self.blocks:
            graph.add_node(b)
        for b in self.blocks.values():
            for out, pops in b.outgoing_jumps.items():
                graph.add_edge(b.offset, out, pops)
        graph.set_entry_point(min(self.blocks))
        graph.process()
        self.graph = graph
        for b in self.blocks.values():
            for out, pops in b.outgoing_jumps.items():
                self.blocks[out].incoming_jumps[b.offset] = pops
        self.liveblocks = dict(((i, self.blocks[i]) for i in self.graph.nodes()))
        for lastblk in reversed(self.blockseq):
            if lastblk in self.liveblocks:
                break
        else:
            raise AssertionError('No live block that exits!?')
        backbone = self.graph.backbone()
        inloopblocks = set()
        for b in self.blocks.keys():
            if self.graph.in_loops(b):
                inloopblocks.add(b)
        self.backbone = backbone - inloopblocks

    def jump(self, target, pops=0):
        """
        Register a jump (conditional or not) to *target* offset.
        *pops* is the number of stack pops implied by the jump (default 0).
        """
        self._curblock.outgoing_jumps[target] = pops

    def _iter_inst(self):
        for inst in self.bytecode:
            if self._use_new_block(inst):
                self._guard_with_as(inst)
                self._start_new_block(inst)
            self._curblock.body.append(inst.offset)
            yield inst

    def _use_new_block(self, inst):
        if inst.offset in self.bytecode.labels:
            res = True
        elif inst.opname in NEW_BLOCKERS:
            res = True
        else:
            res = self._force_new_block
        self._force_new_block = False
        return res

    def _start_new_block(self, inst):
        self._curblock = CFBlock(inst.offset)
        self.blocks[inst.offset] = self._curblock
        self.blockseq.append(inst.offset)

    def _guard_with_as(self, current_inst):
        """Checks if the next instruction after a SETUP_WITH is something other
        than a POP_TOP, if it is something else it'll be some sort of store
        which is not supported (this corresponds to `with CTXMGR as VAR(S)`)."""
        if current_inst.opname == 'SETUP_WITH':
            next_op = self.bytecode[current_inst.next].opname
            if next_op != 'POP_TOP':
                msg = "The 'with (context manager) as (variable):' construct is not supported."
                raise UnsupportedError(msg)

    def op_SETUP_LOOP(self, inst):
        end = inst.get_jump_target()
        self._blockstack.append(end)
        self._loops.append((inst.offset, end))
        self.jump(inst.next)
        self._force_new_block = True

    def op_SETUP_WITH(self, inst):
        end = inst.get_jump_target()
        self._blockstack.append(end)
        self._withs.append((inst.offset, end))
        self.jump(inst.next)
        self._force_new_block = True

    def op_POP_BLOCK(self, inst):
        self._blockstack.pop()

    def op_FOR_ITER(self, inst):
        self.jump(inst.get_jump_target())
        self.jump(inst.next)
        self._force_new_block = True

    def _op_ABSOLUTE_JUMP_IF(self, inst):
        self.jump(inst.get_jump_target())
        self.jump(inst.next)
        self._force_new_block = True
    op_POP_JUMP_IF_FALSE = _op_ABSOLUTE_JUMP_IF
    op_POP_JUMP_IF_TRUE = _op_ABSOLUTE_JUMP_IF
    op_JUMP_IF_FALSE = _op_ABSOLUTE_JUMP_IF
    op_JUMP_IF_TRUE = _op_ABSOLUTE_JUMP_IF
    op_POP_JUMP_FORWARD_IF_FALSE = _op_ABSOLUTE_JUMP_IF
    op_POP_JUMP_BACKWARD_IF_FALSE = _op_ABSOLUTE_JUMP_IF
    op_POP_JUMP_FORWARD_IF_TRUE = _op_ABSOLUTE_JUMP_IF
    op_POP_JUMP_BACKWARD_IF_TRUE = _op_ABSOLUTE_JUMP_IF

    def _op_ABSOLUTE_JUMP_OR_POP(self, inst):
        self.jump(inst.get_jump_target())
        self.jump(inst.next, pops=1)
        self._force_new_block = True
    op_JUMP_IF_FALSE_OR_POP = _op_ABSOLUTE_JUMP_OR_POP
    op_JUMP_IF_TRUE_OR_POP = _op_ABSOLUTE_JUMP_OR_POP

    def op_JUMP_ABSOLUTE(self, inst):
        self.jump(inst.get_jump_target())
        self._force_new_block = True

    def op_JUMP_FORWARD(self, inst):
        self.jump(inst.get_jump_target())
        self._force_new_block = True
    op_JUMP_BACKWARD = op_JUMP_FORWARD

    def op_RETURN_VALUE(self, inst):
        self._curblock.terminating = True
        self._force_new_block = True
    if PYVERSION in ((3, 12),):

        def op_RETURN_CONST(self, inst):
            self._curblock.terminating = True
            self._force_new_block = True
    elif PYVERSION in ((3, 9), (3, 10), (3, 11)):
        pass
    else:
        raise NotImplementedError(PYVERSION)

    def op_RAISE_VARARGS(self, inst):
        self._curblock.terminating = True
        self._force_new_block = True

    def op_BREAK_LOOP(self, inst):
        self.jump(self._blockstack[-1])
        self._force_new_block = True