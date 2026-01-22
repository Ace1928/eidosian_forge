import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model

    A continuous linear model designed to test every form of
    constraint when collecting duals for a minimization
    objective
    