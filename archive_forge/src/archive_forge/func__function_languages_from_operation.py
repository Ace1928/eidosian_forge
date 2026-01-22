import math
import numbers
from typing import cast, Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Union
import numpy as np
import sympy
from cirq_google.api import v2
from cirq_google.ops import InternalGate
def _function_languages_from_operation(value: v2.program_pb2.Operation) -> Iterator[str]:
    for arg in value.args.values():
        yield from _function_languages_from_arg(arg)