import abc
import enum
from typing import Sequence, Dict, Optional, Mapping, NoReturn, List, Tuple
import os
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.param import _ParamData
from pyomo.core.base.block import _BlockData
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.common.config import document_kwargs_from_configdict
from pyomo.common.errors import ApplicationError
from pyomo.common.deprecation import deprecation_warning
from pyomo.opt.results.results_ import SolverResults as LegacySolverResults
from pyomo.opt.results.solution import Solution as LegacySolution
from pyomo.core.kernel.objective import minimize
from pyomo.core.base import SymbolMap
from pyomo.core.base.label import NumericLabeler
from pyomo.core.staleflag import StaleFlagManager
from pyomo.contrib.solver.config import SolverConfig, PersistentSolverConfig
from pyomo.contrib.solver.util import get_objective
from pyomo.contrib.solver.results import (
class PersistentSolverBase(SolverBase):
    """
    Base class upon which persistent solvers can be built. This inherits the
    methods from the solver base class and adds those methods that are necessary
    for persistent solvers.

    Example usage can be seen in the Gurobi interface.
    """

    @document_kwargs_from_configdict(PersistentSolverConfig())
    @abc.abstractmethod
    def solve(self, model: _BlockData, **kwargs) -> Results:
        super().solve(model, kwargs)

    def is_persistent(self):
        """
        Returns
        -------
        is_persistent: bool
            True if the solver is a persistent solver.
        """
        return True

    def _load_vars(self, vars_to_load: Optional[Sequence[_GeneralVarData]]=None) -> NoReturn:
        """
        Load the solution of the primal variables into the value attribute of the variables.

        Parameters
        ----------
        vars_to_load: list
            A list of the variables whose solution should be loaded. If vars_to_load is None, then the solution
            to all primal variables will be loaded.
        """
        for v, val in self._get_primals(vars_to_load=vars_to_load).items():
            v.set_value(val, skip_validation=True)
        StaleFlagManager.mark_all_as_stale(delayed=True)

    @abc.abstractmethod
    def _get_primals(self, vars_to_load: Optional[Sequence[_GeneralVarData]]=None) -> Mapping[_GeneralVarData, float]:
        """
        Get mapping of variables to primals.

        Parameters
        ----------
        vars_to_load : Optional[Sequence[_GeneralVarData]], optional
            Which vars to be populated into the map. The default is None.

        Returns
        -------
        Mapping[_GeneralVarData, float]
            A map of variables to primals.
        """
        raise NotImplementedError(f'{type(self)} does not support the get_primals method')

    def _get_duals(self, cons_to_load: Optional[Sequence[_GeneralConstraintData]]=None) -> Dict[_GeneralConstraintData, float]:
        """
        Declare sign convention in docstring here.

        Parameters
        ----------
        cons_to_load: list
            A list of the constraints whose duals should be loaded. If cons_to_load is None, then the duals for all
            constraints will be loaded.

        Returns
        -------
        duals: dict
            Maps constraints to dual values
        """
        raise NotImplementedError(f'{type(self)} does not support the get_duals method')

    def _get_reduced_costs(self, vars_to_load: Optional[Sequence[_GeneralVarData]]=None) -> Mapping[_GeneralVarData, float]:
        """
        Parameters
        ----------
        vars_to_load: list
            A list of the variables whose reduced cost should be loaded. If vars_to_load is None, then all reduced costs
            will be loaded.

        Returns
        -------
        reduced_costs: ComponentMap
            Maps variable to reduced cost
        """
        raise NotImplementedError(f'{type(self)} does not support the get_reduced_costs method')

    @abc.abstractmethod
    def set_instance(self, model):
        """
        Set an instance of the model
        """

    @abc.abstractmethod
    def set_objective(self, obj: _GeneralObjectiveData):
        """
        Set current objective for the model
        """

    @abc.abstractmethod
    def add_variables(self, variables: List[_GeneralVarData]):
        """
        Add variables to the model
        """

    @abc.abstractmethod
    def add_parameters(self, params: List[_ParamData]):
        """
        Add parameters to the model
        """

    @abc.abstractmethod
    def add_constraints(self, cons: List[_GeneralConstraintData]):
        """
        Add constraints to the model
        """

    @abc.abstractmethod
    def add_block(self, block: _BlockData):
        """
        Add a block to the model
        """

    @abc.abstractmethod
    def remove_variables(self, variables: List[_GeneralVarData]):
        """
        Remove variables from the model
        """

    @abc.abstractmethod
    def remove_parameters(self, params: List[_ParamData]):
        """
        Remove parameters from the model
        """

    @abc.abstractmethod
    def remove_constraints(self, cons: List[_GeneralConstraintData]):
        """
        Remove constraints from the model
        """

    @abc.abstractmethod
    def remove_block(self, block: _BlockData):
        """
        Remove a block from the model
        """

    @abc.abstractmethod
    def update_variables(self, variables: List[_GeneralVarData]):
        """
        Update variables on the model
        """

    @abc.abstractmethod
    def update_parameters(self):
        """
        Update parameters on the model
        """