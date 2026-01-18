from __future__ import annotations
import re
import os
import typing as T
from ...mesonlib import version_compare
from ...interpreterbase import (
@noKwargs
@noPosargs
def underscorify_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> str:
    return re.sub('[^a-zA-Z0-9]', '_', self.held_object)