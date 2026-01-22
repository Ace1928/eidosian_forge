from __future__ import annotations
import abc
import enum
import typing
from .. import types
@typing.final
class Unary(Expr):
    """A unary expression.

    Args:
        op: The opcode describing which operation is being done.
        operand: The operand of the operation.
        type: The resolved type of the result.
    """
    __slots__ = ('op', 'operand')

    class Op(enum.Enum):
        """Enumeration of the opcodes for unary operations.

        The bitwise negation :data:`BIT_NOT` takes a single bit or an unsigned integer of known
        width, and returns a value of the same type.

        The logical negation :data:`LOGIC_NOT` takes an input that is implicitly coerced to a
        Boolean, and returns a Boolean.
        """
        BIT_NOT = 1
        'Bitwise negation. ``~operand``.'
        LOGIC_NOT = 2
        'Logical negation. ``!operand``.'

        def __str__(self):
            return f'Unary.{super().__str__()}'

        def __repr__(self):
            return f'Unary.{super().__repr__()}'

    def __init__(self, op: Unary.Op, operand: Expr, type: types.Type):
        self.op = op
        self.operand = operand
        self.type = type

    def accept(self, visitor, /):
        return visitor.visit_unary(self)

    def __eq__(self, other):
        return isinstance(other, Unary) and self.type == other.type and (self.op is other.op) and (self.operand == other.operand)

    def __repr__(self):
        return f'Unary({self.op}, {self.operand}, {self.type})'