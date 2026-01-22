import enum
from typing import Optional, List, Union, Iterable, Tuple
class AssignmentStatement(Statement):
    """Assignment of an expression to an l-value."""

    def __init__(self, lvalue: SubscriptedIdentifier, rvalue: Expression):
        self.lvalue = lvalue
        self.rvalue = rvalue