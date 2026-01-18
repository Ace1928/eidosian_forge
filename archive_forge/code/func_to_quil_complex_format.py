import string
from typing import Callable, Dict, Set, Tuple, Union, Any, Optional, List, cast
import numpy as np
import cirq
import cirq_rigetti
from cirq import protocols, value, ops
def to_quil_complex_format(num) -> str:
    """A function for outputting a number to a complex string in QUIL format."""
    cnum = complex(str(num))
    return f'{cnum.real}+{cnum.imag}i'