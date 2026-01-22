from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
class ControlFlow(object):
    """Control-flow graph.

       entry_point ControlBlock entry point for this graph
       exit_point  ControlBlock normal exit point
       block       ControlBlock current block
       blocks      set    children nodes
       entries     set    tracked entries
       loops       list   stack for loop descriptors
       exceptions  list   stack for exception descriptors
       in_try_block  int  track if we're in a try...except or try...finally block
    """

    def __init__(self):
        self.blocks = set()
        self.entries = set()
        self.loops = []
        self.exceptions = []
        self.entry_point = ControlBlock()
        self.exit_point = ExitBlock()
        self.blocks.add(self.exit_point)
        self.block = self.entry_point
        self.in_try_block = 0

    def newblock(self, parent=None):
        """Create floating block linked to `parent` if given.

           NOTE: Block is NOT added to self.blocks
        """
        block = ControlBlock()
        self.blocks.add(block)
        if parent:
            parent.add_child(block)
        return block

    def nextblock(self, parent=None):
        """Create block children block linked to current or `parent` if given.

           NOTE: Block is added to self.blocks
        """
        block = ControlBlock()
        self.blocks.add(block)
        if parent:
            parent.add_child(block)
        elif self.block:
            self.block.add_child(block)
        self.block = block
        return self.block

    def is_tracked(self, entry):
        if entry.is_anonymous:
            return False
        return entry.is_local or entry.is_pyclass_attr or entry.is_arg or entry.from_closure or entry.in_closure or entry.error_on_uninitialized

    def is_statically_assigned(self, entry):
        if entry.is_local and entry.is_variable and (entry.type.is_struct_or_union or entry.type.is_complex or entry.type.is_array or (entry.type.is_cpp_class and (not entry.is_cpp_optional))):
            return True
        return False

    def mark_position(self, node):
        """Mark position, will be used to draw graph nodes."""
        if self.block:
            self.block.positions.add(node.pos[:2])

    def mark_assignment(self, lhs, rhs, entry, rhs_scope=None):
        if self.block and self.is_tracked(entry):
            assignment = NameAssignment(lhs, rhs, entry, rhs_scope=rhs_scope)
            self.block.stats.append(assignment)
            self.block.gen[entry] = assignment
            self.entries.add(entry)

    def mark_argument(self, lhs, rhs, entry):
        if self.block and self.is_tracked(entry):
            assignment = Argument(lhs, rhs, entry)
            self.block.stats.append(assignment)
            self.block.gen[entry] = assignment
            self.entries.add(entry)

    def mark_deletion(self, node, entry):
        if self.block and self.is_tracked(entry):
            assignment = NameDeletion(node, entry)
            self.block.stats.append(assignment)
            self.block.gen[entry] = Uninitialized
            self.entries.add(entry)

    def mark_reference(self, node, entry):
        if self.block and self.is_tracked(entry):
            self.block.stats.append(NameReference(node, entry))
            self.entries.add(entry)

    def normalize(self):
        """Delete unreachable and orphan blocks."""
        queue = {self.entry_point}
        visited = set()
        while queue:
            root = queue.pop()
            visited.add(root)
            for child in root.children:
                if child not in visited:
                    queue.add(child)
        unreachable = self.blocks - visited
        for block in unreachable:
            block.detach()
        visited.remove(self.entry_point)
        for block in visited:
            if block.empty():
                for parent in block.parents:
                    for child in block.children:
                        parent.add_child(child)
                block.detach()
                unreachable.add(block)
        self.blocks -= unreachable

    def initialize(self):
        """Set initial state, map assignments to bits."""
        self.assmts = {}
        bit = 1
        for entry in self.entries:
            assmts = AssignmentList()
            assmts.mask = assmts.bit = bit
            self.assmts[entry] = assmts
            bit <<= 1
        for block in self.blocks:
            for stat in block.stats:
                if isinstance(stat, NameAssignment):
                    stat.bit = bit
                    assmts = self.assmts[stat.entry]
                    assmts.stats.append(stat)
                    assmts.mask |= bit
                    bit <<= 1
        for block in self.blocks:
            for entry, stat in block.gen.items():
                assmts = self.assmts[entry]
                if stat is Uninitialized:
                    block.i_gen |= assmts.bit
                else:
                    block.i_gen |= stat.bit
                block.i_kill |= assmts.mask
            block.i_output = block.i_gen
            for entry in block.bounded:
                block.i_kill |= self.assmts[entry].bit
        for assmts in self.assmts.values():
            self.entry_point.i_gen |= assmts.bit
        self.entry_point.i_output = self.entry_point.i_gen

    def map_one(self, istate, entry):
        ret = set()
        assmts = self.assmts[entry]
        if istate & assmts.bit:
            if self.is_statically_assigned(entry):
                ret.add(StaticAssignment(entry))
            elif entry.from_closure:
                ret.add(Unknown)
            else:
                ret.add(Uninitialized)
        for assmt in assmts.stats:
            if istate & assmt.bit:
                ret.add(assmt)
        return ret

    def reaching_definitions(self):
        """Per-block reaching definitions analysis."""
        dirty = True
        while dirty:
            dirty = False
            for block in self.blocks:
                i_input = 0
                for parent in block.parents:
                    i_input |= parent.i_output
                i_output = i_input & ~block.i_kill | block.i_gen
                if i_output != block.i_output:
                    dirty = True
                block.i_input = i_input
                block.i_output = i_output