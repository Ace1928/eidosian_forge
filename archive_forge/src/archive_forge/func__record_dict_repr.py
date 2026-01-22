import abc
import collections
import io
from typing import (
import numpy as np
import pandas as pd
from cirq import value, ops
from cirq._compat import proper_repr
from cirq.study import resolver
def _record_dict_repr(self):
    """Helper function for use in __repr__ to display the records field."""
    return '{' + ', '.join((f'{k!r}: {proper_repr(v)}' for k, v in self.records.items())) + '}'