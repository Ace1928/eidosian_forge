from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.dependencies import attempt_import, numpy as np
from pyomo.core.base.objective import Objective
from pyomo.core.base.suffix import Suffix
from pyomo.core.expr.visitor import identify_variables
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.subsystems import (
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import CyIpoptNLP
from pyomo.contrib.pynumero.algorithms.solvers.scipy_solvers import (
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.contrib.incidence_analysis.scc_solver import (
class PyomoImplicitFunctionBase(object):
    """A base class defining an API for implicit functions defined using
    Pyomo components. In particular, this is the API required by
    ExternalPyomoModel.

    Implicit functions are defined by two lists of Pyomo VarData and
    one list of Pyomo ConstraintData. The first list of VarData corresponds
    to "variables" defining the outputs of the implicit function.
    The list of ConstraintData are equality constraints that are converged
    to evaluate the implicit function. The second list of VarData are
    variables to be treated as "parameters" or inputs to the implicit
    function.

    """

    def __init__(self, variables, constraints, parameters):
        """
        Arguments
        ---------
        variables: List of VarData
            Variables to be treated as outputs of the implicit function
        constraints: List of ConstraintData
            Constraints that are converged to evaluate the implicit function
        parameters: List of VarData
            Variables to be treated as inputs to the implicit function

        """
        self._variables = variables
        self._constraints = constraints
        self._parameters = parameters
        self._block_variables = variables + parameters
        self._block = create_subsystem_block(constraints, self._block_variables)

    def get_variables(self):
        return self._variables

    def get_constraints(self):
        return self._constraints

    def get_parameters(self):
        return self._parameters

    def get_block(self):
        return self._block

    def set_parameters(self, values):
        """Sets the parameters of the system that defines the implicit
        function.

        This method does not necessarily need to update values of the Pyomo
        variables, as long as the next evaluation of this implicit function
        is consistent with these inputs.

        Arguments
        ---------
        values: NumPy array
            Array of values to set for the "parameter variables" in the order
            they were specified in the constructor

        """
        raise NotImplementedError()

    def evaluate_outputs(self):
        """Returns the values of the variables that are treated as outputs
        of the implicit function

        The returned values do not necessarily need to be the values stored
        in the Pyomo variables, as long as they are consistent with the
        latest parameters that have been set.

        Returns
        -------
        NumPy array
            Array with values corresponding to the "output variables" in
            the order they were specified in the constructor

        """
        raise NotImplementedError()

    def update_pyomo_model(self):
        """Sets values of "parameter variables" and "output variables"
        to the most recent values set or computed in this implicit function

        """
        raise NotImplementedError()