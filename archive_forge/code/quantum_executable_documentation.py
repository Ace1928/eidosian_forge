import abc
import dataclasses
from dataclasses import dataclass
from typing import Union, Tuple, Optional, Sequence, cast, Dict, Any, List, Iterator
import cirq
from cirq import _compat, study
Initialize and normalize the quantum executable group.

        Args:
             executables: A sequence of `cg.QuantumExecutable` which will be frozen into a
                tuple.
        