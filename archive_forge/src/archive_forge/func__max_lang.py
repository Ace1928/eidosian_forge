import math
import numbers
from typing import cast, Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Union
import numpy as np
import sympy
from cirq_google.api import v2
from cirq_google.ops import InternalGate
def _max_lang(langs: Iterable[str]) -> str:
    i = max((LANGUAGE_ORDER.index(e) for e in langs), default=0)
    return LANGUAGE_ORDER[i]