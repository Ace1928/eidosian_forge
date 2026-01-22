import pyomo.common.unittest as unittest
from pyomo.core.base import ConcreteModel, Var, Reals
from pyomo.core.beta.dict_objects import (
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.core.base.expression import _GeneralExpressionData
class _TestActiveComponentDictBase(_TestComponentDictBase):

    def test_activate(self):
        model = self.model
        index = ['a', 1, None, (1,), (1, 2)]
        model.c = self._ctype(((i, self._cdatatype(self._arg())) for i in index))
        self.assertEqual(len(model.c), len(index))
        self.assertEqual(model.c.active, True)
        model.c._active = False
        for i in index:
            model.c[i]._active = False
        self.assertEqual(model.c.active, False)
        for i in index:
            self.assertEqual(model.c[i].active, False)
        model.c.activate()
        self.assertEqual(model.c.active, True)

    def test_activate(self):
        model = self.model
        index = ['a', 1, None, (1,), (1, 2)]
        model.c = self._ctype(((i, self._cdatatype(self._arg())) for i in index))
        self.assertEqual(len(model.c), len(index))
        self.assertEqual(model.c.active, True)
        for i in index:
            self.assertEqual(model.c[i].active, True)
        model.c.deactivate()
        self.assertEqual(model.c.active, False)
        for i in index:
            self.assertEqual(model.c[i].active, False)

    def test_active(self):
        model = self.model
        model.c = self._ctype()
        self.assertEqual(model.c.active, True)
        model.c.deactivate()
        self.assertEqual(model.c.active, False)
        model.c[1] = self._cdatatype(self._arg())
        self.assertEqual(model.c.active, True)