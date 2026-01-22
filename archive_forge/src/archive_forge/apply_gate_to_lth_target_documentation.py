import itertools
from typing import Callable, Sequence, Tuple
import attr
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import unary_iteration_gate
Helper constructor to automatically deduce bitsize attributes.