import dataclasses
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
from cirq import protocols
from cirq.linalg import predicates
@dataclasses.dataclass
class _BuildFromSlicesArgs:
    slices: Tuple[_SliceConfig, ...]
    scale: complex