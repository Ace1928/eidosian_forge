import abc
from typing import Any, Dict, Iterator, Sequence, Type, TYPE_CHECKING, Generic, TypeVar
import numpy as np
from cirq import _compat, ops, value, qis
from cirq.sim import simulator, state_vector, simulator_base
from cirq.protocols import qid_shape
iPython (Jupyter) pretty print.