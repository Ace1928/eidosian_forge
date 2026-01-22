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
class _ExternalGreyBoxModelHelper(object):

    def __init__(self, ex_grey_box_block, vardata_to_idx, con_offset):
        """This helper takes an ExternalGreyBoxModel and provides the residual,
        Jacobian, and Hessian computation mapped to the correct variable space.

        The ExternalGreyBoxModel provides an interface that supports
        equality constraints (pure residuals) and output equations. Let
        u be the inputs, o be the outputs, and x be the full set of
        primal variables from the entire pyomo_nlp.

        With this, the ExternalGreyBoxModel provides the residual
        computations w_eq(u), and w_o(u), as well as the Jacobians,
        Jw_eq(u), and Jw_o(u). This helper provides h(x)=0, where h(x) =
        [h_eq(x); h_o(x)-o] and h_eq(x)=w_eq(Pu*x), and
        h_o(x)=w_o(Pu*x), and Pu is a mapping from the full primal
        variables "x" to the inputs "u".

        It also provides the Jacobian of h w.r.t. x.
           J_h(x) = [Jw_eq(Pu*x); Jw_o(Pu*x)-Po*x]
        where Po is a mapping from the full primal variables "x" to the
        outputs "o".

        """
        self._block = ex_grey_box_block
        self._ex_model = ex_grey_box_block.get_external_model()
        self._n_primals = len(vardata_to_idx)
        n_inputs = len(self._block.inputs)
        assert n_inputs == self._ex_model.n_inputs()
        n_eq_constraints = self._ex_model.n_equality_constraints()
        n_outputs = len(self._block.outputs)
        assert n_outputs == self._ex_model.n_outputs()
        self._inputs_to_primals_map = np.fromiter((vardata_to_idx[v] for v in self._block.inputs.values()), dtype=np.int64, count=n_inputs)
        self._outputs_to_primals_map = np.fromiter((vardata_to_idx[v] for v in self._block.outputs.values()), dtype=np.int64, count=n_outputs)
        if self._ex_model.n_outputs() == 0 and self._ex_model.n_equality_constraints() == 0:
            raise ValueError('ExternalGreyBoxModel has no equality constraints or outputs. It must have at least one or both.')
        self._ex_eq_duals_to_full_map = None
        if n_eq_constraints > 0:
            self._ex_eq_duals_to_full_map = list(range(con_offset, con_offset + n_eq_constraints))
        self._ex_output_duals_to_full_map = None
        if n_outputs > 0:
            self._ex_output_duals_to_full_map = list(range(con_offset + n_eq_constraints, con_offset + n_eq_constraints + n_outputs))
        self._eq_jac_primal_jcol = None
        self._outputs_jac_primal_jcol = None
        self._additional_output_entries_irow = None
        self._additional_output_entries_jcol = None
        self._additional_output_entries_data = None
        self._con_offset = con_offset
        self._eq_hess_jcol = None
        self._eq_hess_irow = None
        self._output_hess_jcol = None
        self._output_hess_irow = None

    def set_primals(self, primals):
        input_values = primals[self._inputs_to_primals_map]
        self._ex_model.set_input_values(input_values)
        self._output_values = primals[self._outputs_to_primals_map]

    def set_duals(self, duals):
        if self._ex_eq_duals_to_full_map is not None:
            duals_eq = duals[self._ex_eq_duals_to_full_map]
            self._ex_model.set_equality_constraint_multipliers(duals_eq)
        if self._ex_output_duals_to_full_map is not None:
            duals_outputs = duals[self._ex_output_duals_to_full_map]
            self._ex_model.set_output_constraint_multipliers(duals_outputs)

    def n_residuals(self):
        return self._ex_model.n_equality_constraints() + self._ex_model.n_outputs()

    def get_residual_scaling(self):
        eq_scaling = self._ex_model.get_equality_constraint_scaling_factors()
        output_con_scaling = self._ex_model.get_output_constraint_scaling_factors()
        if eq_scaling is None and output_con_scaling is None:
            return None
        if eq_scaling is None:
            eq_scaling = np.ones(self._ex_model.n_equality_constraints())
        if output_con_scaling is None:
            output_con_scaling = np.ones(self._ex_model.n_outputs())
        return np.concatenate((eq_scaling, output_con_scaling))

    def evaluate_residuals(self):
        resid_list = []
        if self._ex_model.n_equality_constraints() > 0:
            resid_list.append(self._ex_model.evaluate_equality_constraints())
        if self._ex_model.n_outputs() > 0:
            computed_output_values = self._ex_model.evaluate_outputs()
            output_resid = computed_output_values - self._output_values
            resid_list.append(output_resid)
        return np.concatenate(resid_list)

    def evaluate_jacobian(self):
        eq_jac = None
        if self._ex_model.n_equality_constraints() > 0:
            eq_jac = self._ex_model.evaluate_jacobian_equality_constraints()
            if self._eq_jac_primal_jcol is None:
                self._eq_jac_primal_jcol = self._inputs_to_primals_map[eq_jac.col]
            eq_jac = coo_matrix((eq_jac.data, (eq_jac.row, self._eq_jac_primal_jcol)), (eq_jac.shape[0], self._n_primals))
        outputs_jac = None
        if self._ex_model.n_outputs() > 0:
            outputs_jac = self._ex_model.evaluate_jacobian_outputs()
            row = outputs_jac.row
            if self._outputs_jac_primal_jcol is None:
                self._outputs_jac_primal_jcol = self._inputs_to_primals_map[outputs_jac.col]
                self._additional_output_entries_irow = np.asarray(range(self._ex_model.n_outputs()))
                self._additional_output_entries_jcol = self._outputs_to_primals_map
                self._additional_output_entries_data = -1.0 * np.ones(self._ex_model.n_outputs())
            col = self._outputs_jac_primal_jcol
            data = outputs_jac.data
            row = np.concatenate((row, self._additional_output_entries_irow))
            col = np.concatenate((col, self._additional_output_entries_jcol))
            data = np.concatenate((data, self._additional_output_entries_data))
            outputs_jac = coo_matrix((data, (row, col)), shape=(outputs_jac.shape[0], self._n_primals))
        jac = None
        if eq_jac is not None:
            if outputs_jac is not None:
                jac = BlockMatrix(2, 1)
                jac.name = 'external model jacobian'
                jac.set_block(0, 0, eq_jac)
                jac.set_block(1, 0, outputs_jac)
            else:
                assert self._ex_model.n_outputs() == 0
                assert self._ex_model.n_equality_constraints() > 0
                jac = eq_jac
        else:
            assert outputs_jac is not None
            assert self._ex_model.n_outputs() > 0
            assert self._ex_model.n_equality_constraints() == 0
            jac = outputs_jac
        return jac

    def evaluate_hessian(self):
        data_list = []
        irow_list = []
        jcol_list = []
        eq_hess = None
        if self._ex_model.n_equality_constraints() > 0:
            eq_hess = self._ex_model.evaluate_hessian_equality_constraints()
            if self._eq_hess_jcol is None:
                if np.any(eq_hess.row < eq_hess.col):
                    raise ValueError('ExternalGreyBoxModel must return lower triangular portion of the Hessian only')
                self._eq_hess_irow = row = self._inputs_to_primals_map[eq_hess.row]
                self._eq_hess_jcol = col = self._inputs_to_primals_map[eq_hess.col]
                mask = col > row
                row[mask], col[mask] = (col[mask], row[mask])
            data_list.append(eq_hess.data)
            irow_list.append(self._eq_hess_irow)
            jcol_list.append(self._eq_hess_jcol)
        if self._ex_model.n_outputs() > 0:
            output_hess = self._ex_model.evaluate_hessian_outputs()
            if self._output_hess_irow is None:
                if np.any(output_hess.row < output_hess.col):
                    raise ValueError('ExternalGreyBoxModel must return lower triangular portion of the Hessian only')
                self._output_hess_irow = row = self._inputs_to_primals_map[output_hess.row]
                self._output_hess_jcol = col = self._inputs_to_primals_map[output_hess.col]
                mask = col > row
                row[mask], col[mask] = (col[mask], row[mask])
            data_list.append(output_hess.data)
            irow_list.append(self._output_hess_irow)
            jcol_list.append(self._output_hess_jcol)
        data = np.concatenate(data_list)
        irow = np.concatenate(irow_list)
        jcol = np.concatenate(jcol_list)
        hess = coo_matrix((data, (irow, jcol)), (self._n_primals, self._n_primals))
        return hess