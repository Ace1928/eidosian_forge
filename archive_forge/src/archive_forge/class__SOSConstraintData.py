import sys
import logging
from pyomo.common.deprecation import RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.component import ActiveComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import (
from pyomo.core.base.set_types import PositiveIntegers
class _SOSConstraintData(ActiveComponentData):
    """
    This class defines the data for a single special ordered set.

    Constructor arguments:
        owner           The Constraint object that owns this data.

    Public class attributes:
        active          A boolean that is true if this objective is active in the model.
        component       The constraint component.

    Private class attributes:
        _variables       SOS variables.
        _weights         SOS variable weights.
        _level           SOS level (Positive Integer)
    """
    __slots__ = ('_variables', '_weights', '_level')

    def __init__(self, owner):
        """Constructor"""
        self._level = None
        self._variables = []
        self._weights = []
        ActiveComponentData.__init__(self, owner)

    def num_variables(self):
        return len(self._variables)

    def items(self):
        return zip(self._variables, self._weights)

    @property
    def level(self):
        """
        Return the SOS level
        """
        return self._level

    @level.setter
    def level(self, level):
        if level not in PositiveIntegers:
            raise ValueError('SOS Constraint level must be a positive integer')
        self._level = level

    @property
    def variables(self):
        """
        Return the variable list for the SOS constraint
        """
        return self._variables

    def get_variables(self):
        for val in self._variables:
            yield val

    def get_items(self):
        assert len(self._variables) == len(self._weights)
        for v, w in zip(self._variables, self._weights):
            yield (v, w)

    def set_items(self, variables, weights):
        self._variables = []
        self._weights = []
        for v, w in zip(variables, weights):
            self._variables.append(v)
            if w < 0.0:
                raise ValueError('Cannot set negative weight %f for variable %s' % (w, v.name))
            self._weights.append(w)