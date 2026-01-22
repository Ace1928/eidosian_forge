import abc
import logging
from scipy.sparse import coo_matrix
from pyomo.common.dependencies import numpy as np
from pyomo.common.deprecation import RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base import Var, Set, Constraint, value
from pyomo.core.base.block import _BlockData, Block, declare_custom_block
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.initializer import Initializer
from pyomo.core.base.set import UnindexedComponent_set
from pyomo.core.base.reference import Reference
from ..sparse.block_matrix import BlockMatrix
class ExternalGreyBoxBlockData(_BlockData):

    def set_external_model(self, external_grey_box_model, inputs=None, outputs=None):
        """
        Parameters
        ----------
        external_grey_box_model: ExternalGreyBoxModel
            The external model that will be interfaced to in this block
        inputs: List of VarData objects
            If provided, these VarData will be used as inputs into the
            external model.
        outputs: List of VarData objects
            If provided, these VarData will be used as outputs from the
            external model.

        """
        self._ex_model = ex_model = external_grey_box_model
        if ex_model is None:
            self._input_names = self._output_names = None
            self.inputs = self.outputs = None
            return
        self._input_names = ex_model.input_names()
        if self._input_names is None or len(self._input_names) == 0:
            raise ValueError('No input_names specified for external_grey_box_model. Must specify at least one input.')
        self._input_names_set = Set(initialize=self._input_names, ordered=True)
        if inputs is None:
            self.inputs = Var(self._input_names_set)
        else:
            if ex_model.n_inputs() != len(inputs):
                raise ValueError('Dimension mismatch in provided input vars for external model.\nExpected %s input vars, got %s.' % (ex_model.n_inputs(), len(inputs)))
            self.inputs = Reference(inputs)
        self._equality_constraint_names = ex_model.equality_constraint_names()
        self._output_names = ex_model.output_names()
        self._output_names_set = Set(initialize=self._output_names, ordered=True)
        if outputs is None:
            self.outputs = Var(self._output_names_set)
        else:
            if ex_model.n_outputs() != len(outputs):
                raise ValueError('Dimension mismatch in provided output vars for external model.\nExpected %s output vars, got %s.' % (ex_model.n_outputs(), len(outputs)))
            self.outputs = Reference(outputs)
        external_grey_box_model.finalize_block_construction(self)

    def get_external_model(self):
        return self._ex_model