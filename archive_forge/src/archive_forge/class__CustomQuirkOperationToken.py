from typing import (
import cmath
import re
import numpy as np
import sympy
class _CustomQuirkOperationToken:

    def __init__(self, unary_action: Optional[Callable[[_ResolvedToken], _ResolvedToken]], binary_action: Optional[Callable[[_ResolvedToken, _ResolvedToken], _ResolvedToken]], priority: float):
        self.unary_action = unary_action
        self.binary_action = binary_action
        self.priority = priority