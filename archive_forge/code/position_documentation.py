from __future__ import annotations
import typing
from abc import ABC
from copy import copy
from warnings import warn
import numpy as np
from .._utils import check_required_aesthetics, groupby_apply
from .._utils.registry import Register, Registry
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.aes import X_AESTHETICS, Y_AESTHETICS

            Compute function helper
            