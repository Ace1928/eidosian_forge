from typing import AbstractSet, Any, cast, Dict, Optional, Sequence, Tuple, Union
import math
import numbers
import numpy as np
import sympy
import cirq
from cirq import value, protocols
from cirq._compat import proper_repr
from cirq.ops import common_gates, raw_types
from cirq.type_workarounds import NotImplementedType
See `cirq.SupportsCircuitDiagramInfo`.