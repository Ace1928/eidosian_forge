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