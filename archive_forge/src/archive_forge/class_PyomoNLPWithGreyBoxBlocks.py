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
class PyomoNLPWithGreyBoxBlocks(NLP):

    def __init__(self, pyomo_model):
        super(PyomoNLPWithGreyBoxBlocks, self).__init__()
        greybox_components = []
        self._pyomo_model_var_names_to_datas = None
        try:
            for greybox in pyomo_model.component_objects(ExternalGreyBoxBlock, descend_into=True):
                greybox.parent_block().reclassify_component_type(greybox, pyo.Block)
                greybox_components.append(greybox)
            self._pyomo_model = pyomo_model
            self._pyomo_nlp = PyomoNLP(pyomo_model)
            self._pyomo_model_var_names_to_datas = {v.getname(fully_qualified=True): v for v in pyomo_model.component_data_objects(ctype=pyo.Var, descend_into=True)}
            self._pyomo_model_constraint_names_to_datas = {c.getname(fully_qualified=True): c for c in pyomo_model.component_data_objects(ctype=pyo.Constraint, descend_into=True)}
        finally:
            for greybox in greybox_components:
                greybox.parent_block().reclassify_component_type(greybox, ExternalGreyBoxBlock)
        if self._pyomo_nlp.n_primals() == 0:
            raise ValueError('No variables were found in the Pyomo part of the model. PyomoGreyBoxModel requires at least one variable to be active in a Pyomo objective or constraint')
        greybox_nlps = []
        fixed_vars = []
        for greybox in greybox_components:
            for data in greybox.values():
                if data.active:
                    fixed_vars.extend((v for v in data.inputs.values() if v.fixed))
                    fixed_vars.extend((v for v in data.outputs.values() if v.fixed))
                    greybox_nlp = _ExternalGreyBoxAsNLP(data)
                    greybox_nlps.append(greybox_nlp)
        if fixed_vars:
            logging.getLogger(__name__).error('PyomoNLPWithGreyBoxBlocks found fixed variables for the inputs and/or outputs of an ExternalGreyBoxBlock. This is not currently supported. The fixed variables were:\n\t' + '\n\t'.join((f.getname(fully_qualified=True) for f in fixed_vars)))
            raise NotImplementedError('PyomoNLPWithGreyBoxBlocks does not support fixed inputs or outputs')
        primals_names = set(self._pyomo_nlp.primals_names())
        for gbnlp in greybox_nlps:
            primals_names.update(gbnlp.primals_names())
        self._n_primals = len(primals_names)
        self._primals_names = primals_names = sorted(primals_names)
        self._pyomo_model_var_datas = [self._pyomo_model_var_names_to_datas[nm] for nm in self._primals_names]
        self._constraint_names = list(self._pyomo_nlp.constraint_names())
        self._constraint_datas = [self._pyomo_model_constraint_names_to_datas.get(nm) for nm in self._constraint_names]
        for gbnlp in greybox_nlps:
            self._constraint_names.extend(gbnlp.constraint_names())
            self._constraint_datas.extend([(gbnlp._block, nm) for nm in gbnlp.constraint_names()])
        self._n_constraints = len(self._constraint_names)
        self._has_hessian_support = True
        for nlp in greybox_nlps:
            if not nlp.has_hessian_support():
                self._has_hessian_support = False
        self._pyomo_nlp = ProjectedNLP(self._pyomo_nlp, primals_names)
        for i, gbnlp in enumerate(greybox_nlps):
            greybox_nlps[i] = ProjectedNLP(greybox_nlps[i], primals_names)
        self._nlps = nlps = [self._pyomo_nlp]
        nlps.extend(greybox_nlps)
        self._init_primals = self._pyomo_nlp.init_primals()
        self._primals_lb = self._pyomo_nlp.primals_lb()
        self._primals_ub = self._pyomo_nlp.primals_ub()
        for gbnlp in greybox_nlps:
            local = gbnlp.init_primals()
            mask = ~np.isnan(local)
            self._init_primals[mask] = local[mask]
            local = gbnlp.primals_lb()
            mask = ~np.isnan(local)
            self._primals_lb[mask] = np.maximum(self._primals_lb[mask], local[mask])
            local = gbnlp.primals_ub()
            mask = ~np.isnan(local)
            self._primals_ub[mask] = np.minimum(self._primals_ub[mask], local[mask])
        if np.any(np.isnan(self._init_primals)) or np.any(np.isnan(self._primals_lb)) or np.any(np.isnan(self._primals_ub)):
            raise ValueError('NaN values found in initialization of primals or primals_lb or primals_ub in _PyomoNLPWithGreyBoxBlocks.')
        self._init_duals = BlockVector(len(nlps))
        self._dual_values_blockvector = BlockVector(len(nlps))
        self._constraints_lb = BlockVector(len(nlps))
        self._constraints_ub = BlockVector(len(nlps))
        for i, nlp in enumerate(nlps):
            self._init_duals.set_block(i, nlp.init_duals())
            self._constraints_lb.set_block(i, nlp.constraints_lb())
            self._constraints_ub.set_block(i, nlp.constraints_ub())
            self._dual_values_blockvector.set_block(i, np.nan * np.zeros(nlp.n_constraints()))
        self._init_duals = self._init_duals.flatten()
        self._constraints_lb = self._constraints_lb.flatten()
        self._constraints_ub = self._constraints_ub.flatten()
        if np.any(np.isnan(self._init_duals)) or np.any(np.isnan(self._constraints_lb)) or np.any(np.isnan(self._constraints_ub)):
            raise ValueError('NaN values found in initialization of duals or constraints_lb or constraints_ub in _PyomoNLPWithGreyBoxBlocks.')
        self._primal_values = np.nan * np.ones(self._n_primals)
        self.set_primals(self._init_primals)
        self.set_duals(self._init_duals)
        assert not np.any(np.isnan(self._primal_values))
        assert not np.any(np.isnan(self._dual_values_blockvector))
        need_scaling = False
        self._obj_scaling = self._pyomo_nlp.get_obj_scaling()
        if self._obj_scaling is None:
            self._obj_scaling = 1.0
        else:
            need_scaling = True
        self._primals_scaling = np.ones(self.n_primals())
        scaling_suffix = pyomo_model.component('scaling_factor')
        if scaling_suffix and scaling_suffix.ctype is pyo.Suffix:
            need_scaling = True
            for i, v in enumerate(self._pyomo_model_var_datas):
                if v in scaling_suffix:
                    self._primals_scaling[i] = scaling_suffix[v]
        self._constraints_scaling = BlockVector(len(nlps))
        for i, nlp in enumerate(nlps):
            local_constraints_scaling = nlp.get_constraints_scaling()
            if local_constraints_scaling is None:
                self._constraints_scaling.set_block(i, np.ones(nlp.n_constraints()))
            else:
                self._constraints_scaling.set_block(i, local_constraints_scaling)
                need_scaling = True
        if need_scaling:
            self._constraints_scaling = self._constraints_scaling.flatten()
        else:
            self._obj_scaling = None
            self._primals_scaling = None
            self._constraints_scaling = None
        jac = self.evaluate_jacobian()
        self._nnz_jacobian = len(jac.data)
        self._sparse_hessian_summation = None
        self._nnz_hessian_lag = None
        if self._has_hessian_support:
            hess = self.evaluate_hessian_lag()
            self._nnz_hessian_lag = len(hess.data)

    def n_primals(self):
        return self._n_primals

    def primals_names(self):
        return self._primals_names

    def n_constraints(self):
        return self._n_constraints

    def constraint_names(self):
        return self._constraint_names

    def nnz_jacobian(self):
        return self._nnz_jacobian

    def nnz_hessian_lag(self):
        return self._nnz_hessian_lag

    def primals_lb(self):
        return self._primals_lb

    def primals_ub(self):
        return self._primals_ub

    def constraints_lb(self):
        return self._constraints_lb

    def constraints_ub(self):
        return self._constraints_ub

    def init_primals(self):
        return self._init_primals

    def init_duals(self):
        return self._init_duals

    def create_new_vector(self, vector_type):
        if vector_type == 'primals':
            return np.zeros(self.n_primals(), dtype=np.float64)
        elif vector_type == 'constraints' or vector_type == 'duals':
            return np.zeros(self.n_constraints(), dtype=np.float64)
        else:
            raise RuntimeError('Called create_new_vector with an unknown vector_type')

    def set_primals(self, primals):
        np.copyto(self._primal_values, primals)
        for nlp in self._nlps:
            nlp.set_primals(primals)

    def get_primals(self):
        return np.copy(self._primal_values)

    def set_duals(self, duals):
        self._dual_values_blockvector.copyfrom(duals)
        for i, nlp in enumerate(self._nlps):
            nlp.set_duals(self._dual_values_blockvector.get_block(i))

    def get_duals(self):
        return self._dual_values_blockvector.flatten()

    def set_obj_factor(self, obj_factor):
        self._pyomo_nlp.set_obj_factor(obj_factor)

    def get_obj_factor(self):
        return self._pyomo_nlp.get_obj_factor()

    def get_obj_scaling(self):
        return self._obj_scaling

    def get_primals_scaling(self):
        return self._primals_scaling

    def get_constraints_scaling(self):
        return self._constraints_scaling

    def evaluate_objective(self):
        return self._pyomo_nlp.evaluate_objective()

    def evaluate_grad_objective(self, out=None):
        return self._pyomo_nlp.evaluate_grad_objective(out=out)

    def evaluate_constraints(self, out=None):
        ret = BlockVector(len(self._nlps))
        for i, nlp in enumerate(self._nlps):
            ret.set_block(i, nlp.evaluate_constraints())
        if out is not None:
            ret.copyto(out)
            return out
        return ret.flatten()

    def evaluate_jacobian(self, out=None):
        ret = BlockMatrix(len(self._nlps), 1)
        for i, nlp in enumerate(self._nlps):
            ret.set_block(i, 0, nlp.evaluate_jacobian())
        ret = ret.tocoo()
        if out is not None:
            assert np.array_equal(ret.row, out.row)
            assert np.array_equal(ret.col, out.col)
            np.copyto(out.data, ret.data)
            return out
        return ret

    def evaluate_hessian_lag(self, out=None):
        list_of_hessians = [nlp.evaluate_hessian_lag() for nlp in self._nlps]
        if self._sparse_hessian_summation is None:
            self._sparse_hessian_summation = CondensedSparseSummation(list_of_hessians)
        ret = self._sparse_hessian_summation.sum(list_of_hessians)
        if out is not None:
            assert np.array_equal(ret.row, out.row)
            assert np.array_equal(ret.col, out.col)
            np.copyto(out.data, ret.data)
            return out
        return ret

    def report_solver_status(self, status_code, status_message):
        raise NotImplementedError('This is not yet implemented.')

    def load_state_into_pyomo(self, bound_multipliers=None):
        primals = self.get_primals()
        for value, vardata in zip(primals, self._pyomo_model_var_datas):
            vardata.set_value(value)
        m = self._pyomo_model
        model_suffixes = dict(pyo.suffix.active_import_suffix_generator(m))
        obj_sign = 1.0
        objs = list(m.component_data_objects(ctype=pyo.Objective, active=True, descend_into=True))
        assert len(objs) == 1
        if objs[0].sense == pyo.maximize:
            obj_sign = -1.0
        if 'dual' in model_suffixes:
            model_suffixes['dual'].clear()
            dual_values = self._dual_values_blockvector.flatten()
            for value, t in zip(dual_values, self._constraint_datas):
                if type(t) is tuple:
                    model_suffixes['dual'].setdefault(t[0], {})[t[1]] = -obj_sign * value
                else:
                    model_suffixes['dual'][t] = -obj_sign * value
        if 'ipopt_zL_out' in model_suffixes:
            model_suffixes['ipopt_zL_out'].clear()
            if bound_multipliers is not None:
                model_suffixes['ipopt_zL_out'].update(zip(self._pyomo_model_var_datas, obj_sign * bound_multipliers[0]))
        if 'ipopt_zU_out' in model_suffixes:
            model_suffixes['ipopt_zU_out'].clear()
            if bound_multipliers is not None:
                model_suffixes['ipopt_zU_out'].update(zip(self._pyomo_model_var_datas, -obj_sign * bound_multipliers[1]))