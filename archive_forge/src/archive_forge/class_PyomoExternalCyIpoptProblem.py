import numpy as np
import abc
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import (
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from pyomo.environ import Var, Constraint, value
from pyomo.core.base.var import _VarData
from pyomo.common.modeling import unique_component_name
class PyomoExternalCyIpoptProblem(CyIpoptProblemInterface):

    def __init__(self, pyomo_model, ex_input_output_model, inputs, outputs, outputs_eqn_scaling=None, nl_file_options=None):
        """
        Create an instance of this class to pass as a problem to CyIpopt.

        Parameters
        ----------
        pyomo_model : ConcreteModel
           The ConcreteModel representing the Pyomo part of the problem. This
           model must contain Pyomo variables for the inputs and the outputs.

        ex_input_output_model : ExternalInputOutputModel
           An instance of a derived class (from ExternalInputOutputModel) that provides
           the methods to compute the outputs and the derivatives.

        inputs : list of Pyomo variables (_VarData)
           The Pyomo model needs to have variables to represent the inputs to the
           external model. This is the list of those input variables in the order
           that corresponds to the input_values vector provided in the set_inputs call.

        outputs : list of Pyomo variables (_VarData)
          The Pyomo model needs to have variables to represent the outputs from the
          external model. This is the list of those output variables in the order
          that corresponds to the numpy array returned from the evaluate_outputs call.

        outputs_eqn_scaling : list or array-like or None
          This sets the value of scaling parameters for the additional
          output equations that are generated. No scaling is done if this
          is set to None.
        """
        self._pyomo_model = pyomo_model
        self._ex_io_model = ex_input_output_model
        self._inputs = [v for v in inputs]
        for v in self._inputs:
            if not isinstance(v, _VarData):
                raise RuntimeError('Argument inputs passed to PyomoExternalCyIpoptProblem must be a list of VarData objects. Note: if you have an indexed variable, pass each index as a separate entry in the list (e.g., inputs=[m.x[1], m.x[2]]).')
        self._outputs = [v for v in outputs]
        for v in self._outputs:
            if not isinstance(v, _VarData):
                raise RuntimeError('Argument outputs passed to PyomoExternalCyIpoptProblem must be a list of VarData objects. Note: if you have an indexed variable, pass each index as a separate entry in the list (e.g., inputs=[m.x[1], m.x[2]]).')
        dummy_var_name = unique_component_name(self._pyomo_model, '_dummy_variable_CyIpoptPyomoExNLP')
        dummy_var = Var()
        setattr(self._pyomo_model, dummy_var_name, dummy_var)
        dummy_con_name = unique_component_name(self._pyomo_model, '_dummy_constraint_CyIpoptPyomoExNLP')
        dummy_con = Constraint(expr=getattr(self._pyomo_model, dummy_var_name) == sum((v for v in self._inputs)) + sum((v for v in self._outputs)))
        setattr(self._pyomo_model, dummy_con_name, dummy_con)
        dummy_var_value = 0
        for v in self._inputs:
            if v.value is not None:
                dummy_var_value += value(v)
        for v in self._outputs:
            if v.value is not None:
                dummy_var_value += value(v)
        dummy_var.value = dummy_var_value
        self._pyomo_nlp = PyomoNLP(self._pyomo_model, nl_file_options)
        init_primals = self._pyomo_nlp.init_primals()
        init_duals_pyomo = self._pyomo_nlp.init_duals()
        if np.any(np.isnan(init_duals_pyomo)):
            init_duals_pyomo[np.isnan(init_duals_pyomo)] = 1.0
        init_duals_ex = np.ones(len(self._outputs), dtype=np.float64)
        init_duals = BlockVector(2)
        init_duals.set_block(0, init_duals_pyomo)
        init_duals.set_block(1, init_duals_ex)
        self._input_columns = self._pyomo_nlp.get_primal_indices(self._inputs)
        self._output_columns = self._pyomo_nlp.get_primal_indices(self._outputs)
        self._cached_primals = init_primals.copy()
        self._cached_duals = init_duals.clone(copy=True)
        self._cached_obj_factor = 1.0
        self._pyomo_nlp.set_primals(self._cached_primals)
        self._pyomo_nlp.set_duals(self._cached_duals.get_block(0))
        ex_inputs = self._ex_io_inputs_from_full_primals(self._cached_primals)
        self._ex_io_model.set_inputs(ex_inputs)
        pyomo_nlp_con_lb = self._pyomo_nlp.constraints_lb()
        ex_con_lb = np.zeros(len(self._outputs), dtype=np.float64)
        self._gL = np.concatenate((pyomo_nlp_con_lb, ex_con_lb))
        pyomo_nlp_con_ub = self._pyomo_nlp.constraints_ub()
        ex_con_ub = np.zeros(len(self._outputs), dtype=np.float64)
        self._gU = np.concatenate((pyomo_nlp_con_ub, ex_con_ub))
        self._obj_scaling = self._pyomo_nlp.get_obj_scaling()
        self._primals_scaling = self._pyomo_nlp.get_primals_scaling()
        pyomo_constraints_scaling = self._pyomo_nlp.get_constraints_scaling()
        self._constraints_scaling = None
        if pyomo_constraints_scaling is not None or outputs_eqn_scaling is not None:
            if pyomo_constraints_scaling is None:
                pyomo_constraints_scaling = np.ones(self._pyomo_nlp.n_primals(), dtype=np.float64)
            if outputs_eqn_scaling is None:
                outputs_eqn_scaling = np.ones(len(self._outputs), dtype=np.float64)
            if type(outputs_eqn_scaling) is list:
                outputs_eqn_scaling = np.asarray(outputs_eqn_scaling, dtype=np.float64)
            self._constraints_scaling = np.concatenate((pyomo_constraints_scaling, outputs_eqn_scaling))
        self._jac_pyomo = self._pyomo_nlp.evaluate_jacobian()
        ex_start_row = self._pyomo_nlp.n_constraints()
        jac_ex = self._ex_io_model.evaluate_derivatives()
        jac_ex_irows = np.copy(jac_ex.row)
        jac_ex_irows += ex_start_row
        jac_ex_jcols = np.copy(jac_ex.col)
        for z, col in enumerate(jac_ex_jcols):
            jac_ex_jcols[z] = self._input_columns[col]
        jac_ex_data = np.copy(jac_ex.data)
        jac_ex_output_irows = list()
        jac_ex_output_jcols = list()
        jac_ex_output_data = list()
        for i in range(len(self._outputs)):
            jac_ex_output_irows.append(ex_start_row + i)
            jac_ex_output_jcols.append(self._output_columns[i])
            jac_ex_output_data.append(-1.0)
        self._full_jac_irows = np.concatenate((self._jac_pyomo.row, jac_ex_irows, jac_ex_output_irows))
        self._full_jac_jcols = np.concatenate((self._jac_pyomo.col, jac_ex_jcols, jac_ex_output_jcols))
        self._full_jac_data = np.concatenate((self._jac_pyomo.data, jac_ex_data, jac_ex_output_data))
        super(PyomoExternalCyIpoptProblem, self).__init__()

    def load_x_into_pyomo(self, primals):
        """
        Use this method to load a numpy array of values into the corresponding
        Pyomo variables (e.g., the solution from CyIpopt)

        Parameters
        ----------
        primals : numpy array
           The array of values that will be given to the Pyomo variables. The
           order of this array is the same as the order in the PyomoNLP created
           internally.
        """
        pyomo_variables = self._pyomo_nlp.get_pyomo_variables()
        for i, v in enumerate(primals):
            pyomo_variables[i].set_value(v)

    def _set_primals_if_necessary(self, primals):
        if not np.array_equal(primals, self._cached_primals):
            self._pyomo_nlp.set_primals(primals)
            ex_inputs = self._ex_io_inputs_from_full_primals(primals)
            self._ex_io_model.set_inputs(ex_inputs)
            self._cached_primals = primals.copy()

    def _set_duals_if_necessary(self, duals):
        if not np.array_equal(duals, self._cached_duals):
            self._cached_duals.copy_from(duals)
            self._pyomo_nlp.set_duals(self._cached_duals.get_block(0))

    def _set_obj_factor_if_necessary(self, obj_factor):
        if obj_factor != self._cached_obj_factor:
            self._pyomo_nlp.set_obj_factor(obj_factor)
            self._cached_obj_factor = obj_factor

    def x_init(self):
        return self._pyomo_nlp.init_primals()

    def x_lb(self):
        return self._pyomo_nlp.primals_lb()

    def x_ub(self):
        return self._pyomo_nlp.primals_ub()

    def g_lb(self):
        return self._gL.copy()

    def g_ub(self):
        return self._gU.copy()

    def scaling_factors(self):
        return (self._obj_scaling, self._primals_scaling, self._constraints_scaling)

    def objective(self, primals):
        self._set_primals_if_necessary(primals)
        return self._pyomo_nlp.evaluate_objective()

    def gradient(self, primals):
        self._set_primals_if_necessary(primals)
        return self._pyomo_nlp.evaluate_grad_objective()

    def constraints(self, primals):
        self._set_primals_if_necessary(primals)
        pyomo_constraints = self._pyomo_nlp.evaluate_constraints()
        ex_io_outputs = self._ex_io_model.evaluate_outputs()
        ex_io_constraints = ex_io_outputs - self._ex_io_outputs_from_full_primals(primals)
        constraints = BlockVector(2)
        constraints.set_block(0, pyomo_constraints)
        constraints.set_block(1, ex_io_constraints)
        return constraints.flatten()

    def jacobianstructure(self):
        return (self._full_jac_irows, self._full_jac_jcols)

    def jacobian(self, primals):
        self._set_primals_if_necessary(primals)
        self._pyomo_nlp.evaluate_jacobian(out=self._jac_pyomo)
        pyomo_data = self._jac_pyomo.data
        ex_io_deriv = self._ex_io_model.evaluate_derivatives()
        self._full_jac_data[0:len(pyomo_data)] = pyomo_data
        self._full_jac_data[len(pyomo_data):len(pyomo_data) + len(ex_io_deriv.data)] = ex_io_deriv.data
        return self._full_jac_data

    def hessianstructure(self):
        return (np.zeros(0), np.zeros(0))

    def hessian(self, x, y, obj_factor):
        raise NotImplementedError('No Hessians for now')

    def _ex_io_inputs_from_full_primals(self, primals):
        return primals[self._input_columns]

    def _ex_io_outputs_from_full_primals(self, primals):
        return primals[self._output_columns]