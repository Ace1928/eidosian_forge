from __future__ import annotations
import warnings
from contextlib import suppress
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from .._utils import get_valid_kwargs
from ..exceptions import PlotnineError, PlotnineWarning
def _to_patsy_env(environment: Environment) -> EvalEnvironment:
    """
    Convert a plotnine environment to a patsy environment
    """
    from patsy.eval import EvalEnvironment
    eval_env = EvalEnvironment(environment.namespaces)
    return eval_env