import unittest
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import captured_stdout
@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
class TestMonteCarlo(CUDATestCase):
    """
    Test monte-carlo integration
    """

    def setUp(self):
        self._captured_stdout = captured_stdout()
        self._captured_stdout.__enter__()
        super().setUp()

    def tearDown(self):
        self._captured_stdout.__exit__(None, None, None)
        super().tearDown()

    def test_ex_montecarlo(self):
        import numba
        import numpy as np
        from numba import cuda
        from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
        nsamps = 1000000

        @cuda.jit
        def mc_integrator_kernel(out, rng_states, lower_lim, upper_lim):
            """
            kernel to draw random samples and evaluate the function to
            be integrated at those sample values
            """
            size = len(out)
            gid = cuda.grid(1)
            if gid < size:
                samp = xoroshiro128p_uniform_float32(rng_states, gid)
                samp = samp * (upper_lim - lower_lim) + lower_lim
                y = func(samp)
                out[gid] = y

        @cuda.reduce
        def sum_reduce(a, b):
            return a + b

        def mc_integrate(lower_lim, upper_lim, nsamps):
            """
            approximate the definite integral of `func` from
            `lower_lim` to `upper_lim`
            """
            out = cuda.to_device(np.zeros(nsamps, dtype='float32'))
            rng_states = create_xoroshiro128p_states(nsamps, seed=42)
            mc_integrator_kernel.forall(nsamps)(out, rng_states, lower_lim, upper_lim)
            factor = (upper_lim - lower_lim) / (nsamps - 1)
            return sum_reduce(out) * factor

        @numba.jit
        def func(x):
            return 1.0 / x
        mc_integrate(1, 2, nsamps)
        mc_integrate(2, 3, nsamps)
        np.testing.assert_allclose(mc_integrate(1, 2, nsamps), 0.69315, atol=0.001)
        np.testing.assert_allclose(mc_integrate(2, 3, nsamps), 0.4055, atol=0.001)