from typing import (
import cmath
import re
import numpy as np
import sympy
class _HangingNode:

    def __init__(self, func: Callable[[_ResolvedToken, _ResolvedToken], _ResolvedToken], weight: float):
        self.func = func
        self.weight = weight