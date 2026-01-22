import enum
from typing import Optional, List, Union, Iterable, Tuple
class Designator(ASTNode):
    """
    designator
        : LBRACKET expression RBRACKET
    """

    def __init__(self, expression: Expression):
        self.expression = expression