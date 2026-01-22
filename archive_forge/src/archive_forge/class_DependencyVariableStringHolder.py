from __future__ import annotations
import re
import os
import typing as T
from ...mesonlib import version_compare
from ...interpreterbase import (
class DependencyVariableStringHolder(StringHolder):

    def op_div(self, other: str) -> T.Union[str, DependencyVariableString]:
        ret = super().op_div(other)
        if '..' in other:
            return ret
        return DependencyVariableString(ret)