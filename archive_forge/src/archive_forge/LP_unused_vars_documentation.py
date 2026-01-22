import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model

    A continuous linear model where some vars aren't used
    and some used vars start out with the stale flag as True
    