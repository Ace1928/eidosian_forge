import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
def warmstart_model(self):
    assert self.model is not None
    model = self.model
    model.x_unused.value = -1.0
    model.x_unused_initially_stale.value = -1.0
    model.x_unused_initially_stale.stale = True
    for i in model.s:
        model.X_unused[i].value = -1.0
        model.X_unused_initially_stale[i].value = -1.0
        model.X_unused_initially_stale[i].stale = True
    model.x.value = -1.0
    model.x_initially_stale.value = -1.0
    model.x_initially_stale.stale = True
    for i in model.s:
        model.X[i].value = -1.0
        model.X_initially_stale[i].value = -1.0
        model.X_initially_stale[i].stale = True