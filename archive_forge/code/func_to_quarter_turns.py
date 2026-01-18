import math
from typing import List, Optional, Tuple
import numpy as np
import sympy
from cirq import ops, linalg, protocols
from cirq.linalg.tolerance import near_zero_mod
def to_quarter_turns(half_turns):
    return round(2 * half_turns) % 4