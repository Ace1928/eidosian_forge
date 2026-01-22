from pyomo.opt.base.solvers import LegacySolverFactory
from pyomo.common.factory import Factory
from pyomo.contrib.solver.base import LegacySolverWrapper
class SolverFactoryClass(Factory):

    def register(self, name, legacy_name=None, doc=None):
        if legacy_name is None:
            legacy_name = name

        def decorator(cls):
            self._cls[name] = cls
            self._doc[name] = doc

            class LegacySolver(LegacySolverWrapper, cls):
                pass
            LegacySolverFactory.register(legacy_name, doc)(LegacySolver)
            return cls
        return decorator