import abc
import collections
import io
from typing import (
import numpy as np
import pandas as pd
from cirq import value, ops
from cirq._compat import proper_repr
from cirq.study import resolver
def _bitstring(vals: Iterable[Any]) -> str:
    str_list = [str(int(v)) for v in vals]
    separator = '' if all((len(s) == 1 for s in str_list)) else ' '
    return separator.join(str_list)