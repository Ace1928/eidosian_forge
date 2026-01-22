import io
import logging
import sys
from collections.abc import Sequence
from typing import Optional, List, TextIO
from pyomo.common.config import (
from pyomo.common.log import LogStream
from pyomo.common.numeric_types import native_logical_types
from pyomo.common.timing import HierarchicalTimer
class BranchAndBoundConfig(SolverConfig):
    """
    Base config for all direct MIP solver interfaces

    Attributes
    ----------
    rel_gap: float
        The relative value of the gap in relation to the best bound
    abs_gap: float
        The absolute value of the difference between the incumbent and best bound
    """

    def __init__(self, description=None, doc=None, implicit=False, implicit_domain=None, visibility=0):
        super().__init__(description=description, doc=doc, implicit=implicit, implicit_domain=implicit_domain, visibility=visibility)
        self.rel_gap: Optional[float] = self.declare('rel_gap', ConfigValue(domain=NonNegativeFloat, description='Optional termination condition; the relative value of the gap in relation to the best bound'))
        self.abs_gap: Optional[float] = self.declare('abs_gap', ConfigValue(domain=NonNegativeFloat, description='Optional termination condition; the absolute value of the difference between the incumbent and best bound'))