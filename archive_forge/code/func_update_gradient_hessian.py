import warnings
from abc import ABC, abstractmethod
import numpy as np
import scipy.linalg
import scipy.optimize
from ..._loss.loss import HalfSquaredError
from ...exceptions import ConvergenceWarning
from ...utils.optimize import _check_optimize_result
from .._linear_loss import LinearModelLoss
def update_gradient_hessian(self, X, y, sample_weight):
    _, _, self.hessian_warning = self.linear_loss.gradient_hessian(coef=self.coef, X=X, y=y, sample_weight=sample_weight, l2_reg_strength=self.l2_reg_strength, n_threads=self.n_threads, gradient_out=self.gradient, hessian_out=self.hessian, raw_prediction=self.raw_prediction)