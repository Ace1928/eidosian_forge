import os
import numpy as np
import logging
from scipy.sparse import coo_matrix, identity
from pyomo.common.deprecation import deprecated
import pyomo.core.base as pyo
from pyomo.common.collections import ComponentMap
from pyomo.contrib.pynumero.sparse.block_matrix import BlockMatrix
from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from pyomo.contrib.pynumero.interfaces.nlp import NLP
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.utils import (
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.interfaces.nlp_projections import ProjectedNLP
class _ExternalGreyBoxAsNLP(NLP):
    """
    This class takes an ExternalGreyBoxModel and makes it look
    like an NLP so it can be used with other interfaces. Currently,
    the ExternalGreyBoxModel supports constraints only (no objective),
    so some of the methods are not appropriate and raise exceptions
    """

    def __init__(self, external_grey_box_block):
        self._block = external_grey_box_block
        self._ex_model = external_grey_box_block.get_external_model()
        n_inputs = len(self._block.inputs)
        assert n_inputs == self._ex_model.n_inputs()
        n_eq_constraints = self._ex_model.n_equality_constraints()
        n_outputs = len(self._block.outputs)
        assert n_outputs == self._ex_model.n_outputs()
        if self._ex_model.n_outputs() == 0 and self._ex_model.n_equality_constraints() == 0:
            raise ValueError('ExternalGreyBoxModel has no equality constraints or outputs. To use _ExternalGreyBoxAsNLP, it must have at least one or both.')
        self._primals_names = [self._block.inputs[k].getname(fully_qualified=True) for k in self._block.inputs]
        self._primals_names.extend((self._block.outputs[k].getname(fully_qualified=True) for k in self._block.outputs))
        n_primals = len(self._primals_names)
        prefix = self._block.getname(fully_qualified=True)
        self._constraint_names = ['{}.{}'.format(prefix, nm) for nm in self._ex_model.equality_constraint_names()]
        output_var_names = [self._block.outputs[k].getname(fully_qualified=False) for k in self._block.outputs]
        self._constraint_names.extend(['{}.output_constraints[{}]'.format(prefix, nm) for nm in self._ex_model.output_names()])
        self._primals_lb = BlockVector(2)
        self._primals_ub = BlockVector(2)
        self._init_primals = BlockVector(2)
        lb = np.nan * np.zeros(n_inputs)
        ub = np.nan * np.zeros(n_inputs)
        init_primals = np.nan * np.zeros(n_inputs)
        for i, k in enumerate(self._block.inputs):
            lb[i] = _default_if_none(self._block.inputs[k].lb, -np.inf)
            ub[i] = _default_if_none(self._block.inputs[k].ub, np.inf)
            init_primals[i] = _default_if_none(self._block.inputs[k].value, 0.0)
        self._primals_lb.set_block(0, lb)
        self._primals_ub.set_block(0, ub)
        self._init_primals.set_block(0, init_primals)
        lb = np.nan * np.zeros(n_outputs)
        ub = np.nan * np.zeros(n_outputs)
        init_primals = np.nan * np.zeros(n_outputs)
        for i, k in enumerate(self._block.outputs):
            lb[i] = _default_if_none(self._block.outputs[k].lb, -np.inf)
            ub[i] = _default_if_none(self._block.outputs[k].ub, np.inf)
            init_primals[i] = _default_if_none(self._block.outputs[k].value, 0.0)
        self._primals_lb.set_block(1, lb)
        self._primals_ub.set_block(1, ub)
        self._init_primals.set_block(1, init_primals)
        self._primals_lb = self._primals_lb.flatten()
        self._primals_ub = self._primals_ub.flatten()
        self._init_primals = self._init_primals.flatten()
        self._primal_values = np.copy(self._init_primals)
        self.set_primals(self._init_primals)
        self._init_duals = np.zeros(self.n_constraints(), dtype=np.float64)
        self._dual_values = np.copy(self._init_duals)
        self.set_duals(self._init_duals)
        self._constraints_lb = np.zeros(self.n_constraints(), dtype=np.float64)
        self._constraints_ub = np.zeros(self.n_constraints(), dtype=np.float64)
        self._has_hessian_support = True
        if self._ex_model.n_equality_constraints() > 0 and (not hasattr(self._ex_model, 'evaluate_hessian_equality_constraints')):
            self._has_hessian_support = False
        if self._ex_model.n_outputs() > 0 and (not hasattr(self._ex_model, 'evaluate_hessian_outputs')):
            self._has_hessian_support = False
        self._nnz_jacobian = None
        self._nnz_hessian_lag = None
        self._cached_constraint_residuals = None
        self._cached_jacobian = None
        self._cached_hessian = None

    def n_primals(self):
        return len(self._primals_names)

    def primals_names(self):
        return list(self._primals_names)

    def n_constraints(self):
        return len(self._constraint_names)

    def constraint_names(self):
        return list(self._constraint_names)

    def nnz_jacobian(self):
        if self._nnz_jacobian is None:
            J = self.evaluate_jacobian()
            self._nnz_jacobian = len(J.data)
        return self._nnz_jacobian

    def nnz_hessian_lag(self):
        if self._nnz_hessian_lag is None:
            H = self.evaluate_hessian_lag()
            self._nnz_hessian_lag = len(H.data)
        return self._nnz_hessian_lag

    def primals_lb(self):
        return np.copy(self._primals_lb)

    def primals_ub(self):
        return np.copy(self._primals_ub)

    def constraints_lb(self):
        return np.copy(self._constraints_lb)

    def constraints_ub(self):
        return np.copy(self._constraints_ub)

    def init_primals(self):
        return np.copy(self._init_primals)

    def init_duals(self):
        return np.copy(self._init_duals)

    def create_new_vector(self, vector_type):
        if vector_type == 'primals':
            return np.zeros(self.n_primals(), dtype=np.float64)
        elif vector_type == 'constraints' or vector_type == 'duals':
            return np.zeros(self.n_constraints(), dtype=np.float64)

    def _cache_invalidate_primals(self):
        self._cached_constraint_residuals = None
        self._cached_jacobian = None
        self._cached_hessian = None

    def set_primals(self, primals):
        self._cache_invalidate_primals()
        assert len(primals) == self.n_primals()
        np.copyto(self._primal_values, primals)
        self._ex_model.set_input_values(primals[:self._ex_model.n_inputs()])

    def get_primals(self):
        return np.copy(self._primal_values)

    def _cache_invalidate_duals(self):
        self._cached_hessian = None

    def set_duals(self, duals):
        self._cache_invalidate_duals()
        assert len(duals) == self.n_constraints()
        np.copyto(self._dual_values, duals)
        if self._ex_model.n_equality_constraints() > 0:
            self._ex_model.set_equality_constraint_multipliers(self._dual_values[:self._ex_model.n_equality_constraints()])
        if self._ex_model.n_outputs() > 0:
            self._ex_model.set_output_constraint_multipliers(self._dual_values[self._ex_model.n_equality_constraints():])

    def get_duals(self):
        return np.copy(self._dual_values)

    def set_obj_factor(self, obj_factor):
        raise NotImplementedError('_ExternalGreyBoxAsNLP does not support objectives')

    def get_obj_factor(self):
        raise NotImplementedError('_ExternalGreyBoxAsNLP does not support objectives')

    def get_obj_scaling(self):
        raise NotImplementedError('_ExternalGreyBoxAsNLP does not support objectives')

    def get_primals_scaling(self):
        raise NotImplementedError('_ExternalGreyBoxAsNLP does not support scaling of primals directly. This should be handled at a higher level using suffixes on the Pyomo variables.')

    def get_constraints_scaling(self):
        scaling = np.ones(self.n_constraints(), dtype=np.float64)
        scaled = False
        if self._ex_model.n_equality_constraints() > 0:
            eq_scaling = self._ex_model.get_equality_constraint_scaling_factors()
            if eq_scaling is not None:
                scaling[:self._ex_model.n_equality_constraints()] = eq_scaling
                scaled = True
        if self._ex_model.n_outputs() > 0:
            output_scaling = self._ex_model.get_output_constraint_scaling_factors()
            if output_scaling is not None:
                scaling[self._ex_model.n_equality_constraints():] = output_scaling
                scaled = True
        if scaled:
            return scaling
        return None

    def evaluate_objective(self):
        raise NotImplementedError('_ExternalGreyBoxNLP does not support objectives')

    def evaluate_grad_objective(self, out=None):
        raise NotImplementedError('_ExternalGreyBoxNLP does not support objectives')

    def _evaluate_constraints_if_necessary_and_cache(self):
        if self._cached_constraint_residuals is None:
            c = BlockVector(2)
            if self._ex_model.n_equality_constraints() > 0:
                c.set_block(0, self._ex_model.evaluate_equality_constraints())
            else:
                c.set_block(0, np.zeros(0, dtype=np.float64))
            if self._ex_model.n_outputs() > 0:
                output_values = self._primal_values[self._ex_model.n_inputs():]
                c.set_block(1, self._ex_model.evaluate_outputs() - output_values)
            else:
                c.set_block(1, np.zeros(0, dtype=np.float64))
            self._cached_constraint_residuals = c.flatten()

    def evaluate_constraints(self, out=None):
        self._evaluate_constraints_if_necessary_and_cache()
        if out is not None:
            assert len(out) == self.n_constraints()
            np.copyto(out, self._cached_constraint_residuals)
            return out
        return np.copy(self._cached_constraint_residuals)

    def _evaluate_jacobian_if_necessary_and_cache(self):
        if self._cached_jacobian is None:
            jac = BlockMatrix(2, 2)
            jac.set_row_size(0, self._ex_model.n_equality_constraints())
            jac.set_row_size(1, self._ex_model.n_outputs())
            jac.set_col_size(0, self._ex_model.n_inputs())
            jac.set_col_size(1, self._ex_model.n_outputs())
            if self._ex_model.n_equality_constraints() > 0:
                jac.set_block(0, 0, self._ex_model.evaluate_jacobian_equality_constraints())
            if self._ex_model.n_outputs() > 0:
                jac.set_block(1, 0, self._ex_model.evaluate_jacobian_outputs())
                jac.set_block(1, 1, -1.0 * identity(self._ex_model.n_outputs()))
            self._cached_jacobian = jac.tocoo()

    def evaluate_jacobian(self, out=None):
        self._evaluate_jacobian_if_necessary_and_cache()
        if out is not None:
            jac = self._cached_jacobian
            assert np.array_equal(jac.row, out.row)
            assert np.array_equal(jac.col, out.col)
            np.copyto(out.data, jac.data)
            return out
        return self._cached_jacobian.copy()

    def _evaluate_hessian_if_necessary_and_cache(self):
        if self._cached_hessian is None:
            hess = BlockMatrix(2, 2)
            hess.set_row_size(0, self._ex_model.n_inputs())
            hess.set_row_size(1, self._ex_model.n_outputs())
            hess.set_col_size(0, self._ex_model.n_inputs())
            hess.set_col_size(1, self._ex_model.n_outputs())
            eq_hess = None
            if self._ex_model.n_equality_constraints() > 0:
                eq_hess = self._ex_model.evaluate_hessian_equality_constraints()
                if np.any(eq_hess.row < eq_hess.col):
                    raise ValueError('ExternalGreyBoxModel must return lower triangular portion of the Hessian only')
                eq_hess = make_lower_triangular_full(eq_hess)
            output_hess = None
            if self._ex_model.n_outputs() > 0:
                output_hess = self._ex_model.evaluate_hessian_outputs()
                if np.any(output_hess.row < output_hess.col):
                    raise ValueError('ExternalGreyBoxModel must return lower triangular portion of the Hessian only')
                output_hess = make_lower_triangular_full(output_hess)
            input_hess = None
            if eq_hess is not None and output_hess is not None:
                row = np.concatenate((eq_hess.row, output_hess.row))
                col = np.concatenate((eq_hess.col, output_hess.col))
                data = np.concatenate((eq_hess.data, output_hess.data))
                assert eq_hess.shape == output_hess.shape
                input_hess = coo_matrix((data, (row, col)), shape=eq_hess.shape)
            elif eq_hess is not None:
                input_hess = eq_hess
            elif output_hess is not None:
                input_hess = output_hess
            assert input_hess is not None
            hess.set_block(0, 0, input_hess)
            self._cached_hessian = hess.tocoo()

    def has_hessian_support(self):
        return self._has_hessian_support

    def evaluate_hessian_lag(self, out=None):
        if not self._has_hessian_support:
            raise NotImplementedError('Hessians not supported for all of the external grey box models. Therefore, Hessians are not supported overall.')
        self._evaluate_hessian_if_necessary_and_cache()
        if out is not None:
            hess = self._cached_hessian
            assert np.array_equal(hess.row, out.row)
            assert np.array_equal(hess.col, out.col)
            np.copyto(out.data, hess.data)
            return out
        return self._cached_hessian.copy()

    def report_solver_status(self, status_code, status_message):
        raise NotImplementedError('report_solver_status not implemented')