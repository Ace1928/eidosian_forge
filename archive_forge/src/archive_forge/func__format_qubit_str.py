import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
def _format_qubit_str(qubit: Union[Qubit, QubitPlaceholder, FormalArgument]) -> str:
    if isinstance(qubit, QubitPlaceholder):
        return '{%s}' % str(qubit)
    return str(qubit)