import numbers
import warnings
import numpy as np
from .kalman_smoother import KalmanSmoother
from .cfa_simulation_smoother import CFASimulationSmoother
from . import tools

        Perform simulation smoothing

        Does not return anything, but populates the object's `simulated_*`
        attributes, as specified by simulation output.

        Parameters
        ----------
        simulation_output : int, optional
            Bitmask controlling simulation output. Default is to use the
            simulation output defined in object initialization.
        measurement_disturbance_variates : array_like, optional
            If specified, these are the shocks to the measurement equation,
            :math:`\varepsilon_t`. If unspecified, these are automatically
            generated using a pseudo-random number generator. If specified,
            must be shaped `nsimulations` x `k_endog`, where `k_endog` is the
            same as in the state space model.
        state_disturbance_variates : array_like, optional
            If specified, these are the shocks to the state equation,
            :math:`\eta_t`. If unspecified, these are automatically
            generated using a pseudo-random number generator. If specified,
            must be shaped `nsimulations` x `k_posdef` where `k_posdef` is the
            same as in the state space model.
        initial_state_variates : array_like, optional
            If specified, this is the state vector at time zero, which should
            be shaped (`k_states` x 1), where `k_states` is the same as in the
            state space model. If unspecified, but the model has been
            initialized, then that initialization is used.
        initial_state_variates : array_likes, optional
            Random values to use as initial state variates. Usually only
            specified if results are to be replicated (e.g. to enforce a seed)
            or for testing. If not specified, random variates are drawn.
        pretransformed_measurement_disturbance_variates : bool, optional
            If `measurement_disturbance_variates` is provided, this flag
            indicates whether it should be directly used as the shocks. If
            False, then it is assumed to contain draws from the standard Normal
            distribution that must be transformed using the `obs_cov`
            covariance matrix. Default is False.
        pretransformed_state_disturbance_variates : bool, optional
            If `state_disturbance_variates` is provided, this flag indicates
            whether it should be directly used as the shocks. If False, then it
            is assumed to contain draws from the standard Normal distribution
            that must be transformed using the `state_cov` covariance matrix.
            Default is False.
        pretransformed_initial_state_variates : bool, optional
            If `initial_state_variates` is provided, this flag indicates
            whether it should be directly used as the initial_state. If False,
            then it is assumed to contain draws from the standard Normal
            distribution that must be transformed using the `initial_state_cov`
            covariance matrix. Default is False.
        random_state : {None, int, Generator, RandomState}, optional
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``numpy.random.RandomState`` instance
            is used, seeded with `seed`.
            If `seed` is already a ``numpy.random.Generator`` or
            ``numpy.random.RandomState`` instance then that instance is used.
        disturbance_variates : bool, optional
            Deprecated, please use pretransformed_measurement_shocks and
            pretransformed_state_shocks instead.

            .. deprecated:: 0.14.0

               Use ``measurement_disturbance_variates`` and
               ``state_disturbance_variates`` as replacements.

        pretransformed : bool, optional
            Deprecated, please use pretransformed_measurement_shocks and
            pretransformed_state_shocks instead.

            .. deprecated:: 0.14.0

               Use ``pretransformed_measurement_disturbance_variates`` and
               ``pretransformed_state_disturbance_variates`` as replacements.
        