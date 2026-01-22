import os
import numpy as np
from scipy.sparse import coo_matrix
from pyomo.common.deprecation import deprecated
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import WriterFactory
import pyomo.core.base as pyo
from pyomo.common.collections import ComponentMap
from pyomo.common.env import CtypesEnviron
from ..sparse.block_matrix import BlockMatrix
from pyomo.contrib.pynumero.interfaces.ampl_nlp import AslNLP
from pyomo.contrib.pynumero.interfaces.nlp import NLP
from .external_grey_box import ExternalGreyBoxBlock
class PyomoNLP(AslNLP):

    def __init__(self, pyomo_model, nl_file_options=None):
        """
        Pyomo nonlinear program interface

        Parameters
        ----------
        pyomo_model: pyomo.environ.ConcreteModel
            Pyomo concrete model
        """
        TempfileManager.push()
        try:
            nl_file = TempfileManager.create_tempfile(suffix='pynumero.nl')
            objectives = list(pyomo_model.component_data_objects(ctype=pyo.Objective, active=True, descend_into=True))
            if len(objectives) != 1:
                raise NotImplementedError('The ASL interface and PyomoNLP in PyNumero currently only support single objective problems. Deactivate any extra objectives you may have, or add a dummy objective (f(x)=0) if you have a square problem (found %s objectives).' % (len(objectives),))
            self._objective = objectives[0]
            if nl_file_options is None:
                nl_file_options = dict()
            fname, symbolMap = WriterFactory('nl')(pyomo_model, nl_file, lambda x: True, nl_file_options)
            self._symbol_map = symbolMap
            self._vardata_to_idx = vdidx = ComponentMap()
            self._condata_to_idx = cdidx = ComponentMap()
            for name, obj in symbolMap.bySymbol.items():
                if name[0] == 'v':
                    vdidx[obj] = int(name[1:])
                elif name[0] == 'c':
                    cdidx[obj] = int(name[1:])
            amplfunc = '\n'.join(filter(None, (os.environ.get('AMPLFUNC', None), os.environ.get('PYOMO_AMPLFUNC', None))))
            with CtypesEnviron(AMPLFUNC=amplfunc):
                super(PyomoNLP, self).__init__(nl_file)
            self._pyomo_model = pyomo_model
            full_to_equality = self._con_full_eq_map
            equality_mask = self._con_full_eq_mask
            self._condata_to_eq_idx = ComponentMap(((con, full_to_equality[i]) for con, i in self._condata_to_idx.items() if equality_mask[i]))
            full_to_inequality = self._con_full_ineq_map
            inequality_mask = self._con_full_ineq_mask
            self._condata_to_ineq_idx = ComponentMap(((con, full_to_inequality[i]) for con, i in self._condata_to_idx.items() if inequality_mask[i]))
        finally:
            TempfileManager.pop()

    @property
    def symbol_map(self):
        return self._symbol_map

    def pyomo_model(self):
        """
        Return optimization model
        """
        return self._pyomo_model

    def get_pyomo_objective(self):
        """
        Return an instance of the active objective function on the Pyomo model.
        (there can be only one)
        """
        return self._objective

    def get_pyomo_variables(self):
        """
        Return an ordered list of the Pyomo VarData objects in
        the order corresponding to the primals
        """
        idx_to_vardata = {i: v for v, i in self._vardata_to_idx.items()}
        return [idx_to_vardata[i] for i in range(len(idx_to_vardata))]

    def get_pyomo_constraints(self):
        """
        Return an ordered list of the Pyomo ConData objects in
        the order corresponding to the primals
        """
        idx_to_condata = {i: v for v, i in self._condata_to_idx.items()}
        return [idx_to_condata[i] for i in range(len(idx_to_condata))]

    def get_pyomo_equality_constraints(self):
        """
        Return an ordered list of the Pyomo ConData objects in
        the order corresponding to the equality constraints.
        """
        idx_to_condata = {i: c for c, i in self._condata_to_eq_idx.items()}
        return [idx_to_condata[i] for i in range(len(idx_to_condata))]

    def get_pyomo_inequality_constraints(self):
        """
        Return an ordered list of the Pyomo ConData objects in
        the order corresponding to the inequality constraints.
        """
        idx_to_condata = {i: c for c, i in self._condata_to_ineq_idx.items()}
        return [idx_to_condata[i] for i in range(len(idx_to_condata))]

    @deprecated(msg='This method has been replaced with primals_names', version='6.0.0', remove_in='6.0')
    def variable_names(self):
        return self.primals_names()

    def primals_names(self):
        """
        Return an ordered list of the Pyomo variable
        names in the order corresponding to the primals
        """
        pyomo_variables = self.get_pyomo_variables()
        return [v.getname(fully_qualified=True) for v in pyomo_variables]

    def constraint_names(self):
        """
        Return an ordered list of the Pyomo constraint
        names in the order corresponding to internal constraint order
        """
        pyomo_constraints = self.get_pyomo_constraints()
        return [v.getname(fully_qualified=True) for v in pyomo_constraints]

    def equality_constraint_names(self):
        """
        Return an ordered list of the Pyomo ConData names in
        the order corresponding to the equality constraints.
        """
        equality_constraints = self.get_pyomo_equality_constraints()
        return [v.getname(fully_qualified=True) for v in equality_constraints]

    def inequality_constraint_names(self):
        """
        Return an ordered list of the Pyomo ConData names in
        the order corresponding to the inequality constraints.
        """
        inequality_constraints = self.get_pyomo_inequality_constraints()
        return [v.getname(fully_qualified=True) for v in inequality_constraints]

    def get_primal_indices(self, pyomo_variables):
        """
        Return the list of indices for the primals
        corresponding to the list of Pyomo variables provided

        Parameters
        ----------
        pyomo_variables : list of Pyomo Var or VarData objects
        """
        assert isinstance(pyomo_variables, list)
        var_indices = []
        for v in pyomo_variables:
            if v.is_indexed():
                for vd in v.values():
                    var_id = self._vardata_to_idx[vd]
                    var_indices.append(var_id)
            else:
                var_id = self._vardata_to_idx[v]
                var_indices.append(var_id)
        return var_indices

    def get_constraint_indices(self, pyomo_constraints):
        """
        Return the list of indices for the constraints
        corresponding to the list of Pyomo constraints provided

        Parameters
        ----------
        pyomo_constraints : list of Pyomo Constraint or ConstraintData objects
        """
        assert isinstance(pyomo_constraints, list)
        con_indices = []
        for c in pyomo_constraints:
            if c.is_indexed():
                for cd in c.values():
                    con_id = self._condata_to_idx[cd]
                    con_indices.append(con_id)
            else:
                con_id = self._condata_to_idx[c]
                con_indices.append(con_id)
        return con_indices

    def get_equality_constraint_indices(self, constraints):
        """
        Return the list of equality indices for the constraints
        corresponding to the list of Pyomo constraints provided.

        Parameters
        ----------
        constraints : list of Pyomo Constraints or ConstraintData objects
        """
        indices = []
        for c in constraints:
            if c.is_indexed():
                for cd in c.values():
                    con_eq_idx = self._condata_to_eq_idx[cd]
                    indices.append(con_eq_idx)
            else:
                con_eq_idx = self._condata_to_eq_idx[c]
                indices.append(con_eq_idx)
        return indices

    def get_inequality_constraint_indices(self, constraints):
        """
        Return the list of inequality indices for the constraints
        corresponding to the list of Pyomo constraints provided.

        Parameters
        ----------
        constraints : list of Pyomo Constraints or ConstraintData objects
        """
        indices = []
        for c in constraints:
            if c.is_indexed():
                for cd in c.values():
                    con_ineq_idx = self._condata_to_ineq_idx[cd]
                    indices.append(con_ineq_idx)
            else:
                con_ineq_idx = self._condata_to_ineq_idx[c]
                indices.append(con_ineq_idx)
        return indices

    def get_obj_scaling(self):
        obj = self.get_pyomo_objective()
        scaling_suffix = self._pyomo_model.component('scaling_factor')
        if scaling_suffix and scaling_suffix.ctype is pyo.Suffix:
            if obj in scaling_suffix:
                return scaling_suffix[obj]
            return 1.0
        return None

    def get_primals_scaling(self):
        scaling_suffix = self._pyomo_model.component('scaling_factor')
        if scaling_suffix and scaling_suffix.ctype is pyo.Suffix:
            primals_scaling = np.ones(self.n_primals())
            for i, v in enumerate(self.get_pyomo_variables()):
                if v in scaling_suffix:
                    primals_scaling[i] = scaling_suffix[v]
            return primals_scaling
        return None

    def get_constraints_scaling(self):
        scaling_suffix = self._pyomo_model.component('scaling_factor')
        if scaling_suffix and scaling_suffix.ctype is pyo.Suffix:
            constraints_scaling = np.ones(self.n_constraints())
            for i, c in enumerate(self.get_pyomo_constraints()):
                if c in scaling_suffix:
                    constraints_scaling[i] = scaling_suffix[c]
            return constraints_scaling
        return None

    def extract_subvector_grad_objective(self, pyomo_variables):
        """Compute the gradient of the objective and return the entries
        corresponding to the given Pyomo variables

        Parameters
        ----------
        pyomo_variables : list of Pyomo Var or VarData objects
        """
        grad_obj = self.evaluate_grad_objective()
        return grad_obj[self.get_primal_indices(pyomo_variables)]

    def extract_subvector_constraints(self, pyomo_constraints):
        """
        Return the values of the constraints
        corresponding to the list of Pyomo constraints provided

        Parameters
        ----------
        pyomo_constraints : list of Pyomo Constraint or ConstraintData objects
        """
        residuals = self.evaluate_constraints()
        return residuals[self.get_constraint_indices(pyomo_constraints)]

    def extract_submatrix_jacobian(self, pyomo_variables, pyomo_constraints):
        """
        Return the submatrix of the jacobian that corresponds to the list
        of Pyomo variables and list of Pyomo constraints provided

        Parameters
        ----------
        pyomo_variables : list of Pyomo Var or VarData objects
        pyomo_constraints : list of Pyomo Constraint or ConstraintData objects
        """
        jac = self.evaluate_jacobian()
        primal_indices = self.get_primal_indices(pyomo_variables)
        constraint_indices = self.get_constraint_indices(pyomo_constraints)
        row_mask = np.isin(jac.row, constraint_indices)
        col_mask = np.isin(jac.col, primal_indices)
        submatrix_mask = row_mask & col_mask
        submatrix_irows = np.compress(submatrix_mask, jac.row)
        submatrix_jcols = np.compress(submatrix_mask, jac.col)
        submatrix_data = np.compress(submatrix_mask, jac.data)
        row_submatrix_map = {j: i for i, j in enumerate(constraint_indices)}
        for i, v in enumerate(submatrix_irows):
            submatrix_irows[i] = row_submatrix_map[v]
        col_submatrix_map = {j: i for i, j in enumerate(primal_indices)}
        for i, v in enumerate(submatrix_jcols):
            submatrix_jcols[i] = col_submatrix_map[v]
        return coo_matrix((submatrix_data, (submatrix_irows, submatrix_jcols)), shape=(len(constraint_indices), len(primal_indices)))

    def extract_submatrix_hessian_lag(self, pyomo_variables_rows, pyomo_variables_cols):
        """
        Return the submatrix of the hessian of the lagrangian that
        corresponds to the list of Pyomo variables provided

        Parameters
        ----------
        pyomo_variables_rows : list of Pyomo Var or VarData objects
            List of Pyomo Var or VarData objects corresponding to the desired rows
        pyomo_variables_cols : list of Pyomo Var or VarData objects
            List of Pyomo Var or VarData objects corresponding to the desired columns
        """
        hess_lag = self.evaluate_hessian_lag()
        primal_indices_rows = self.get_primal_indices(pyomo_variables_rows)
        primal_indices_cols = self.get_primal_indices(pyomo_variables_cols)
        row_mask = np.isin(hess_lag.row, primal_indices_rows)
        col_mask = np.isin(hess_lag.col, primal_indices_cols)
        submatrix_mask = row_mask & col_mask
        submatrix_irows = np.compress(submatrix_mask, hess_lag.row)
        submatrix_jcols = np.compress(submatrix_mask, hess_lag.col)
        submatrix_data = np.compress(submatrix_mask, hess_lag.data)
        submatrix_map = {j: i for i, j in enumerate(primal_indices_rows)}
        for i, v in enumerate(submatrix_irows):
            submatrix_irows[i] = submatrix_map[v]
        submatrix_map = {j: i for i, j in enumerate(primal_indices_cols)}
        for i, v in enumerate(submatrix_jcols):
            submatrix_jcols[i] = submatrix_map[v]
        return coo_matrix((submatrix_data, (submatrix_irows, submatrix_jcols)), shape=(len(primal_indices_rows), len(primal_indices_cols)))

    def load_state_into_pyomo(self, bound_multipliers=None):
        primals = self.get_primals()
        variables = self.get_pyomo_variables()
        for var, val in zip(variables, primals):
            var.set_value(val)
        m = self.pyomo_model()
        model_suffixes = dict(pyo.suffix.active_import_suffix_generator(m))
        if 'dual' in model_suffixes:
            duals = self.get_duals()
            constraints = self.get_pyomo_constraints()
            model_suffixes['dual'].clear()
            model_suffixes['dual'].update(zip(constraints, duals))
        if 'ipopt_zL_out' in model_suffixes:
            model_suffixes['ipopt_zL_out'].clear()
            if bound_multipliers is not None:
                model_suffixes['ipopt_zL_out'].update(zip(variables, bound_multipliers[0]))
        if 'ipopt_zU_out' in model_suffixes:
            model_suffixes['ipopt_zU_out'].clear()
            if bound_multipliers is not None:
                model_suffixes['ipopt_zU_out'].update(zip(variables, bound_multipliers[1]))