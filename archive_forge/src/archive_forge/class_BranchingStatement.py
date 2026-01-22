import enum
from typing import Optional, List, Union, Iterable, Tuple
class BranchingStatement(Statement):
    """
    branchingStatement
        : 'if' LPAREN booleanExpression RPAREN programBlock ( 'else' programBlock )?
    """

    def __init__(self, condition: Expression, true_body: ProgramBlock, false_body=None):
        self.condition = condition
        self.true_body = true_body
        self.false_body = false_body