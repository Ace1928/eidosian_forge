import copy
from math import sqrt, log, exp
from itertools import cycle
import warnings
import numpy
from . import tools
def _compute_lambda_parameters(self):
    """Computes the parameters depending on :math:`\\lambda`. It needs to
        be called again if :math:`\\lambda` changes during evolution.
        """
    self.d = self.params.get('d', 1.0 + self.dim / (2.0 * self.lambda_))
    self.ptarg = self.params.get('ptarg', 1.0 / (5 + numpy.sqrt(self.lambda_) / 2.0))
    self.cp = self.params.get('cp', self.ptarg * self.lambda_ / (2 + self.ptarg * self.lambda_))
    self.beta = self.params.get('beta', 0.1 / (self.lambda_ * (self.dim + 2.0)))