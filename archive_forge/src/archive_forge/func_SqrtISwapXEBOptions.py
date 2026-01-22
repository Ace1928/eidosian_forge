import dataclasses
from abc import abstractmethod, ABC
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, ops, protocols, _import
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
def SqrtISwapXEBOptions(*args, **kwargs):
    """Options for calibrating a sqrt(ISWAP) gate using XEB."""
    return XEBPhasedFSimCharacterizationOptions(*args, **kwargs).with_defaults_from_gate(ops.SQRT_ISWAP)