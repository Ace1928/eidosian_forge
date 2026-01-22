from __future__ import annotations
from datetime import datetime
from functools import partial
import operator
from typing import (
import numpy as np
from pandas._libs.tslibs import Timestamp
from pandas.core.dtypes.common import (
import pandas.core.common as com
from pandas.core.computation.common import (
from pandas.core.computation.scope import DEFAULT_GLOBALS
from pandas.io.formats.printing import (
class MathCall(Op):

    def __init__(self, func, args) -> None:
        super().__init__(func.name, args)
        self.func = func

    def __call__(self, env):
        operands = [op(env) for op in self.operands]
        return self.func.func(*operands)

    def __repr__(self) -> str:
        operands = map(str, self.operands)
        return pprint_thing(f'{self.op}({','.join(operands)})')