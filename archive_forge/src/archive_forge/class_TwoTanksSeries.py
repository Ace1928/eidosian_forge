import os
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
import math
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
from pyomo.contrib.pynumero.interfaces.cyipopt_interface import (
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoGreyBoxNLP, PyomoNLP
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
from pyomo.contrib.pynumero.interfaces.tests.compare_utils import (
class TwoTanksSeries(ExternalGreyBoxModel):

    def __init__(self, A1, A2, c1, c2, N, dt):
        self._A1 = A1
        self._A2 = A2
        self._c1 = c1
        self._c2 = c2
        self._N = N
        self._dt = dt
        self._input_names = ['F1_{}'.format(t) for t in range(1, N)]
        self._input_names.extend(['F2_{}'.format(t) for t in range(1, N)])
        self._input_names.extend(['h1_{}'.format(t) for t in range(0, N)])
        self._input_names.extend(['h2_{}'.format(t) for t in range(0, N)])
        self._output_names = ['F12_{}'.format(t) for t in range(0, N)]
        self._output_names.extend(['Fo_{}'.format(t) for t in range(0, N)])
        self._equality_constraint_names = ['h1bal_{}'.format(t) for t in range(1, N)]
        self._equality_constraint_names.extend(['h2bal_{}'.format(t) for t in range(1, N)])
        self._F1 = np.zeros(N)
        self._F2 = np.zeros(N)
        self._h1 = np.zeros(N)
        self._h2 = np.zeros(N)
        self._F12 = np.zeros(N)
        self._Fo = np.zeros(N)
        self._eq_con_mult_values = np.ones(2 * (N - 1))
        self._output_con_mult_values = np.ones(2 * N)

    def input_names(self):
        return self._input_names

    def equality_constraint_names(self):
        return self._equality_constraint_names

    def output_names(self):
        return self._output_names

    def finalize_block_construction(self, pyomo_block):
        for k in pyomo_block.inputs:
            pyomo_block.inputs[k].setlb(0)
            pyomo_block.inputs[k].value = 1.0
        for k in pyomo_block.outputs:
            pyomo_block.outputs[k].setlb(0)
            pyomo_block.outputs[k].value = 1.0

    def set_input_values(self, input_values):
        N = self._N
        assert len(input_values) == 4 * N - 2
        self._F1[1:self._N] = np.copy(input_values[:N - 1])
        self._F2[1:self._N] = np.copy(input_values[N - 1:2 * N - 2])
        self._h1 = np.copy(input_values[2 * N - 2:3 * N - 2])
        self._h2 = np.copy(input_values[3 * N - 2:4 * N - 2])

    def set_equality_constraint_multipliers(self, eq_con_multiplier_values):
        assert len(eq_con_multiplier_values) == 2 * (self._N - 1)
        np.copyto(self._eq_con_mult_values, eq_con_multiplier_values)

    def set_output_constraint_multipliers(self, output_con_multiplier_values):
        assert len(output_con_multiplier_values) == 2 * self._N
        np.copyto(self._output_con_mult_values, output_con_multiplier_values)

    def evaluate_equality_constraints(self):
        N = self._N
        F1 = self._F1
        F2 = self._F2
        h1 = self._h1
        h2 = self._h2
        resid = np.zeros(2 * (N - 1))
        for t in range(1, N):
            resid[t - 1] = h1[t] - h1[t - 1] - self._dt / self._A1 * (F1[t] - self._c1 * math.sqrt(h1[t]))
        for t in range(1, N):
            resid[t - 2 + N] = h2[t] - h2[t - 1] - self._dt / self._A2 * (self._c1 * math.sqrt(h1[t]) + F2[t] - self._c2 * math.sqrt(h2[t]))
        return resid

    def evaluate_outputs(self):
        N = self._N
        h1 = self._h1
        h2 = self._h2
        resid = np.zeros(2 * N)
        for t in range(N):
            resid[t] = self._c1 * math.sqrt(h1[t])
        for t in range(N):
            resid[t + N] = self._c2 * math.sqrt(h2[t])
        return resid

    def evaluate_jacobian_equality_constraints(self):
        N = self._N
        F1 = self._F1
        F2 = self._F2
        h1 = self._h1
        h2 = self._h2
        A1 = self._A1
        A2 = self._A2
        c1 = self._c1
        c2 = self._c2
        dt = self._dt
        nnz = 3 * (N - 1) + 4 * (N - 1)
        irow = np.zeros(nnz, dtype=np.int64)
        jcol = np.zeros(nnz, dtype=np.int64)
        data = np.zeros(nnz, dtype=np.float64)
        idx = 0
        for i in range(N - 1):
            irow[idx] = i
            jcol[idx] = i
            data[idx] = -dt / A1
            idx += 1
            irow[idx] = i
            jcol[idx] = 2 * (N - 1) + i
            data[idx] = -1
            idx += 1
            irow[idx] = i
            jcol[idx] = 2 * (N - 1) + i + 1
            data[idx] = 1 + dt / A1 * c1 * 1 / 2 * h1[i + 1] ** (-0.5)
            idx += 1
        for i in range(N - 1):
            irow[idx] = i + (N - 1)
            jcol[idx] = i + (N - 1)
            data[idx] = -dt / A2
            idx += 1
            irow[idx] = i + (N - 1)
            jcol[idx] = 2 * (N - 1) + i + 1
            data[idx] = -dt / A2 * c1 * 1 / 2 * h1[i + 1] ** (-0.5)
            idx += 1
            irow[idx] = i + (N - 1)
            jcol[idx] = 2 * (N - 1) + N + i
            data[idx] = -1
            idx += 1
            irow[idx] = i + (N - 1)
            jcol[idx] = 2 * (N - 1) + N + i + 1
            data[idx] = 1 + dt / A2 * c2 * 1 / 2 * h2[i + 1] ** (-0.5)
            idx += 1
        assert idx == nnz
        return spa.coo_matrix((data, (irow, jcol)), shape=(2 * (N - 1), 2 * (N - 1) + 2 * N))

    def evaluate_jacobian_outputs(self):
        N = self._N
        F1 = self._F1
        F2 = self._F2
        h1 = self._h1
        h2 = self._h2
        A1 = self._A1
        A2 = self._A2
        c1 = self._c1
        c2 = self._c2
        dt = self._dt
        nnz = 2 * N
        irow = np.zeros(nnz, dtype=np.int64)
        jcol = np.zeros(nnz, dtype=np.int64)
        data = np.zeros(nnz, dtype=np.float64)
        idx = 0
        for i in range(N):
            irow[idx] = i
            jcol[idx] = 2 * (N - 1) + i
            data[idx] = 1 / 2 * c1 * h1[i] ** (-0.5)
            idx += 1
        for i in range(N):
            irow[idx] = N + i
            jcol[idx] = 2 * (N - 1) + N + i
            data[idx] = 1 / 2 * c2 * h2[i] ** (-0.5)
            idx += 1
        assert idx == nnz
        return spa.coo_matrix((data, (irow, jcol)), shape=(2 * N, 2 * (N - 1) + 2 * N))

    def evaluate_hessian_equality_constraints(self):
        N = self._N
        F1 = self._F1
        F2 = self._F2
        h1 = self._h1
        h2 = self._h2
        A1 = self._A1
        A2 = self._A2
        c1 = self._c1
        c2 = self._c2
        dt = self._dt
        lam = self._eq_con_mult_values
        nnz = 2 * (N - 1)
        irow = np.zeros(nnz, dtype=np.int64)
        jcol = np.zeros(nnz, dtype=np.int64)
        data = np.zeros(nnz, dtype=np.float64)
        idx = 0
        for i in range(N - 1):
            irow[idx] = 2 * (N - 1) + i + 1
            jcol[idx] = 2 * (N - 1) + i + 1
            data[idx] = lam[i] * dt / A1 * (-c1 / 4) * h1[i + 1] ** (-1.5) + lam[N - 1 + i] * dt / A2 * (c1 / 4) * h1[i + 1] ** (-1.5)
            idx += 1
            irow[idx] = 2 * (N - 1) + N + i + 1
            jcol[idx] = 2 * (N - 1) + N + i + 1
            data[idx] = lam[N - 1 + i] * dt / A2 * (-c2 / 4) * h2[i + 1] ** (-1.5)
            idx += 1
        assert idx == nnz
        hess = spa.coo_matrix((data, (irow, jcol)), shape=(2 * (N - 1) + 2 * N, 2 * (N - 1) + 2 * N))
        return hess

    def evaluate_hessian_outputs(self):
        N = self._N
        F1 = self._F1
        F2 = self._F2
        h1 = self._h1
        h2 = self._h2
        A1 = self._A1
        A2 = self._A2
        c1 = self._c1
        c2 = self._c2
        dt = self._dt
        lam = self._output_con_mult_values
        nnz = 2 * N
        irow = np.zeros(nnz, dtype=np.int64)
        jcol = np.zeros(nnz, dtype=np.int64)
        data = np.zeros(nnz, dtype=np.float64)
        idx = 0
        for i in range(N):
            irow[idx] = 2 * (N - 1) + i
            jcol[idx] = 2 * (N - 1) + i
            data[idx] = lam[i] * c1 * (-1 / 4) * h1[i] ** (-1.5)
            idx += 1
        for i in range(N):
            irow[idx] = 2 * (N - 1) + N + i
            jcol[idx] = 2 * (N - 1) + N + i
            data[idx] = lam[N + i] * c2 * (-1 / 4) * h2[i] ** (-1.5)
            idx += 1
        assert idx == nnz
        hess = spa.coo_matrix((data, (irow, jcol)), shape=(2 * (N - 1) + 2 * N, 2 * (N - 1) + 2 * N))
        return hess