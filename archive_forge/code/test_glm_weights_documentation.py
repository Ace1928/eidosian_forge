import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_raises
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.datasets.cpunish import load
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.tools import add_constant
from .results import (

Test for weights in GLM, Poisson and OLS/WLS, continuous test_glm.py


Below is a table outlining the test coverage.

================================= ====================== ====== ===================== === ======= ======== ============== ============= ============== ============= ============== ==== =========
Test                              Compared To            params normalized_cov_params bse loglike deviance resid_response resid_pearson resid_deviance resid_working resid_anscombe chi2 optimizer
================================= ====================== ====== ===================== === ======= ======== ============== ============= ============== ============= ============== ==== =========
TestGlmPoissonPlain               stata                  X                            X   X       X        X              X             X              X             X              X    bfgs
TestGlmPoissonFwNr                stata                  X                            X   X       X        X              X             X              X             X              X    bfgs
TestGlmPoissonAwNr                stata                  X                            X   X       X        X              X             X              X             X              X    bfgs
TestGlmPoissonFwHC                stata                  X                            X   X       X                                                                                 X
TestGlmPoissonAwHC                stata                  X                            X   X       X                                                                                 X
TestGlmPoissonFwClu               stata                  X                            X   X       X                                                                                 X
TestGlmTweedieAwNr                R                      X                            X           X        X              X             X              X                                 newton
TestGlmGammaAwNr                  R                      X                            X   special X        X              X             X              X                                 bfgs
TestGlmGaussianAwNr               R                      X                            X   special X        X              X             X              X                                 bfgs
TestRepeatedvsAggregated          statsmodels.GLM        X      X                                                                                                                        bfgs
TestRepeatedvsAverage             statsmodels.GLM        X      X                                                                                                                        bfgs
TestTweedieRepeatedvsAggregated   statsmodels.GLM        X      X                                                                                                                        bfgs
TestTweedieRepeatedvsAverage      statsmodels.GLM        X      X                                                                                                                        bfgs
TestBinomial0RepeatedvsAverage    statsmodels.GLM        X      X
TestBinomial0RepeatedvsDuplicated statsmodels.GLM        X      X                                                                                                                        bfgs
TestBinomialVsVarWeights          statsmodels.GLM        X      X                     X                                                                                                  bfgs
TestGlmGaussianWLS                statsmodels.WLS        X      X                     X                                                                                                  bfgs
================================= ====================== ====== ===================== === ======= ======== ============== ============= ============== ============= ============== ==== =========
