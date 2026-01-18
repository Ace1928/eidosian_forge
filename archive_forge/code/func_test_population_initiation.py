import multiprocessing
import platform
from scipy.optimize._differentialevolution import (DifferentialEvolutionSolver,
from scipy.optimize import differential_evolution, OptimizeResult
from scipy.optimize._constraints import (Bounds, NonlinearConstraint,
from scipy.optimize import rosen, minimize
from scipy.sparse import csr_matrix
from scipy import stats
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_almost_equal,
from pytest import raises as assert_raises, warns
import pytest
def test_population_initiation(self):
    assert_raises(ValueError, DifferentialEvolutionSolver, *(rosen, self.bounds), **{'init': 'rubbish'})
    solver = DifferentialEvolutionSolver(rosen, self.bounds)
    solver.init_population_random()
    assert_equal(solver._nfev, 0)
    assert_(np.all(np.isinf(solver.population_energies)))
    solver.init_population_lhs()
    assert_equal(solver._nfev, 0)
    assert_(np.all(np.isinf(solver.population_energies)))
    solver.init_population_qmc(qmc_engine='halton')
    assert_equal(solver._nfev, 0)
    assert_(np.all(np.isinf(solver.population_energies)))
    solver = DifferentialEvolutionSolver(rosen, self.bounds, init='sobol')
    solver.init_population_qmc(qmc_engine='sobol')
    assert_equal(solver._nfev, 0)
    assert_(np.all(np.isinf(solver.population_energies)))
    population = np.linspace(-1, 3, 10).reshape(5, 2)
    solver = DifferentialEvolutionSolver(rosen, self.bounds, init=population, strategy='best2bin', atol=0.01, seed=1, popsize=5)
    assert_equal(solver._nfev, 0)
    assert_(np.all(np.isinf(solver.population_energies)))
    assert_(solver.num_population_members == 5)
    assert_(solver.population_shape == (5, 2))
    unscaled_population = np.clip(solver._unscale_parameters(population), 0, 1)
    assert_almost_equal(solver.population[:5], unscaled_population)
    assert_almost_equal(np.min(solver.population[:5]), 0)
    assert_almost_equal(np.max(solver.population[:5]), 1)
    population = np.linspace(-1, 3, 15).reshape(5, 3)
    assert_raises(ValueError, DifferentialEvolutionSolver, *(rosen, self.bounds), **{'init': population})
    x0 = np.random.uniform(low=0.0, high=2.0, size=2)
    solver = DifferentialEvolutionSolver(rosen, self.bounds, x0=x0)
    assert_allclose(solver.population[0], x0 / 2.0)