import string
from typing import Callable, Dict, Set, Tuple, Union, Any, Optional, List, cast
import numpy as np
import cirq
import cirq_rigetti
from cirq import protocols, value, ops
Calls `output_func` for successive lines of QUIL output.

        Args:
            output_func: A function that accepts a string of QUIL. This will likely
                write the QUIL to a file.

        Returns:
            None.
        