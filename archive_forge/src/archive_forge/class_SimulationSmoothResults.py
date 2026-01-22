import numbers
import warnings
import numpy as np
from .kalman_smoother import KalmanSmoother
from .cfa_simulation_smoother import CFASimulationSmoother
from . import tools
class SimulationSmoothResults:
    """
    Results from applying the Kalman smoother and/or filter to a state space
    model.

    Parameters
    ----------
    model : Representation
        A Statespace representation
    simulation_smoother : {{prefix}}SimulationSmoother object
        The Cython simulation smoother object with which to simulation smooth.
    random_state : {None, int, Generator, RandomState}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``numpy.random.RandomState`` instance
        is used, seeded with `seed`.
        If `seed` is already a ``numpy.random.Generator`` or
        ``numpy.random.RandomState`` instance then that instance is used.

    Attributes
    ----------
    model : Representation
        A Statespace representation
    dtype : dtype
        Datatype of representation matrices
    prefix : str
        BLAS prefix of representation matrices
    simulation_output : int
        Bitmask controlling simulation output.
    simulate_state : bool
        Flag for if the state is included in simulation output.
    simulate_disturbance : bool
        Flag for if the state and observation disturbances are included in
        simulation output.
    simulate_all : bool
        Flag for if simulation output should include everything.
    generated_measurement_disturbance : ndarray
        Measurement disturbance variates used to genereate the observation
        vector.
    generated_state_disturbance : ndarray
        State disturbance variates used to genereate the state and
        observation vectors.
    generated_obs : ndarray
        Generated observation vector produced as a byproduct of simulation
        smoothing.
    generated_state : ndarray
        Generated state vector produced as a byproduct of simulation smoothing.
    simulated_state : ndarray
        Simulated state.
    simulated_measurement_disturbance : ndarray
        Simulated measurement disturbance.
    simulated_state_disturbance : ndarray
        Simulated state disturbance.
    """

    def __init__(self, model, simulation_smoother, random_state=None):
        self.model = model
        self.prefix = model.prefix
        self.dtype = model.dtype
        self._simulation_smoother = simulation_smoother
        self.random_state = check_random_state(random_state)
        self._generated_measurement_disturbance = None
        self._generated_state_disturbance = None
        self._generated_obs = None
        self._generated_state = None
        self._simulated_state = None
        self._simulated_measurement_disturbance = None
        self._simulated_state_disturbance = None

    @property
    def simulation_output(self):
        return self._simulation_smoother.simulation_output

    @simulation_output.setter
    def simulation_output(self, value):
        self._simulation_smoother.simulation_output = value

    @property
    def simulate_state(self):
        return bool(self.simulation_output & SIMULATION_STATE)

    @simulate_state.setter
    def simulate_state(self, value):
        if bool(value):
            self.simulation_output = self.simulation_output | SIMULATION_STATE
        else:
            self.simulation_output = self.simulation_output & ~SIMULATION_STATE

    @property
    def simulate_disturbance(self):
        return bool(self.simulation_output & SIMULATION_DISTURBANCE)

    @simulate_disturbance.setter
    def simulate_disturbance(self, value):
        if bool(value):
            self.simulation_output = self.simulation_output | SIMULATION_DISTURBANCE
        else:
            self.simulation_output = self.simulation_output & ~SIMULATION_DISTURBANCE

    @property
    def simulate_all(self):
        return bool(self.simulation_output & SIMULATION_ALL)

    @simulate_all.setter
    def simulate_all(self, value):
        if bool(value):
            self.simulation_output = self.simulation_output | SIMULATION_ALL
        else:
            self.simulation_output = self.simulation_output & ~SIMULATION_ALL

    @property
    def generated_measurement_disturbance(self):
        """
        Randomly drawn measurement disturbance variates

        Used to construct `generated_obs`.

        Notes
        -----

        .. math::

           \\varepsilon_t^+ ~ N(0, H_t)

        If `disturbance_variates` were provided to the `simulate()` method,
        then this returns those variates (which were N(0,1)) transformed to the
        distribution above.
        """
        if self._generated_measurement_disturbance is None:
            self._generated_measurement_disturbance = np.array(self._simulation_smoother.measurement_disturbance_variates, copy=True).reshape(self.model.nobs, self.model.k_endog)
        return self._generated_measurement_disturbance

    @property
    def generated_state_disturbance(self):
        """
        Randomly drawn state disturbance variates, used to construct
        `generated_state` and `generated_obs`.

        Notes
        -----

        .. math::

            \\eta_t^+ ~ N(0, Q_t)

        If `disturbance_variates` were provided to the `simulate()` method,
        then this returns those variates (which were N(0,1)) transformed to the
        distribution above.
        """
        if self._generated_state_disturbance is None:
            self._generated_state_disturbance = np.array(self._simulation_smoother.state_disturbance_variates, copy=True).reshape(self.model.nobs, self.model.k_posdef)
        return self._generated_state_disturbance

    @property
    def generated_obs(self):
        """
        Generated vector of observations by iterating on the observation and
        transition equations, given a random initial state draw and random
        disturbance draws.

        Notes
        -----

        .. math::

            y_t^+ = d_t + Z_t \\alpha_t^+ + \\varepsilon_t^+
        """
        if self._generated_obs is None:
            self._generated_obs = np.array(self._simulation_smoother.generated_obs, copy=True)
        return self._generated_obs

    @property
    def generated_state(self):
        """
        Generated vector of states by iterating on the transition equation,
        given a random initial state draw and random disturbance draws.

        Notes
        -----

        .. math::

            \\alpha_{t+1}^+ = c_t + T_t \\alpha_t^+ + \\eta_t^+
        """
        if self._generated_state is None:
            self._generated_state = np.array(self._simulation_smoother.generated_state, copy=True)
        return self._generated_state

    @property
    def simulated_state(self):
        """
        Random draw of the state vector from its conditional distribution.

        Notes
        -----

        .. math::

            \\alpha ~ p(\\alpha \\mid Y_n)
        """
        if self._simulated_state is None:
            self._simulated_state = np.array(self._simulation_smoother.simulated_state, copy=True)
        return self._simulated_state

    @property
    def simulated_measurement_disturbance(self):
        """
        Random draw of the measurement disturbance vector from its conditional
        distribution.

        Notes
        -----

        .. math::

            \\varepsilon ~ N(\\hat \\varepsilon, Var(\\hat \\varepsilon \\mid Y_n))
        """
        if self._simulated_measurement_disturbance is None:
            self._simulated_measurement_disturbance = np.array(self._simulation_smoother.simulated_measurement_disturbance, copy=True)
        return self._simulated_measurement_disturbance

    @property
    def simulated_state_disturbance(self):
        """
        Random draw of the state disturbanc e vector from its conditional
        distribution.

        Notes
        -----

        .. math::

            \\eta ~ N(\\hat \\eta, Var(\\hat \\eta \\mid Y_n))
        """
        if self._simulated_state_disturbance is None:
            self._simulated_state_disturbance = np.array(self._simulation_smoother.simulated_state_disturbance, copy=True)
        return self._simulated_state_disturbance

    def simulate(self, simulation_output=-1, disturbance_variates=None, measurement_disturbance_variates=None, state_disturbance_variates=None, initial_state_variates=None, pretransformed=None, pretransformed_measurement_disturbance_variates=None, pretransformed_state_disturbance_variates=None, pretransformed_initial_state_variates=False, random_state=None):
        """
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
            :math:`\\varepsilon_t`. If unspecified, these are automatically
            generated using a pseudo-random number generator. If specified,
            must be shaped `nsimulations` x `k_endog`, where `k_endog` is the
            same as in the state space model.
        state_disturbance_variates : array_like, optional
            If specified, these are the shocks to the state equation,
            :math:`\\eta_t`. If unspecified, these are automatically
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
        """
        if disturbance_variates is not None:
            msg = '`disturbance_variates` keyword is deprecated, use `measurement_disturbance_variates` and `state_disturbance_variates` instead.'
            warnings.warn(msg, FutureWarning)
            if measurement_disturbance_variates is not None or state_disturbance_variates is not None:
                raise ValueError('Cannot use `disturbance_variates` in combination with  `measurement_disturbance_variates` or `state_disturbance_variates`.')
            if disturbance_variates is not None:
                disturbance_variates = disturbance_variates.ravel()
                n_mds = self.model.nobs * self.model.k_endog
                measurement_disturbance_variates = disturbance_variates[:n_mds]
                state_disturbance_variates = disturbance_variates[n_mds:]
        if pretransformed is not None:
            msg = '`pretransformed` keyword is deprecated, use `pretransformed_measurement_disturbance_variates` and `pretransformed_state_disturbance_variates` instead.'
            warnings.warn(msg, FutureWarning)
            if pretransformed_measurement_disturbance_variates is not None or pretransformed_state_disturbance_variates is not None:
                raise ValueError('Cannot use `pretransformed` in combination with  `pretransformed_measurement_disturbance_variates` or `pretransformed_state_disturbance_variates`.')
            if pretransformed is not None:
                pretransformed_measurement_disturbance_variates = pretransformed
                pretransformed_state_disturbance_variates = pretransformed
        if pretransformed_measurement_disturbance_variates is None:
            pretransformed_measurement_disturbance_variates = False
        if pretransformed_state_disturbance_variates is None:
            pretransformed_state_disturbance_variates = False
        self._generated_measurement_disturbance = None
        self._generated_state_disturbance = None
        self._generated_state = None
        self._generated_obs = None
        self._generated_state = None
        self._simulated_state = None
        self._simulated_measurement_disturbance = None
        self._simulated_state_disturbance = None
        if random_state is None:
            random_state = self.random_state
        else:
            random_state = check_random_state(random_state)
        prefix, dtype, create_smoother, create_filter, create_statespace = self.model._initialize_smoother()
        if create_statespace:
            raise ValueError('The simulation smoother currently cannot replace the underlying _{{prefix}}Representation model object if it changes (which happens e.g. if the dimensions of some system matrices change.')
        self.model._initialize_state(prefix=prefix)
        if measurement_disturbance_variates is not None:
            self._simulation_smoother.set_measurement_disturbance_variates(np.array(measurement_disturbance_variates, dtype=self.dtype).ravel(), pretransformed=pretransformed_measurement_disturbance_variates)
        else:
            self._simulation_smoother.draw_measurement_disturbance_variates(random_state)
        if state_disturbance_variates is not None:
            self._simulation_smoother.set_state_disturbance_variates(np.array(state_disturbance_variates, dtype=self.dtype).ravel(), pretransformed=pretransformed_state_disturbance_variates)
        else:
            self._simulation_smoother.draw_state_disturbance_variates(random_state)
        if initial_state_variates is not None:
            if pretransformed_initial_state_variates:
                self._simulation_smoother.set_initial_state(np.array(initial_state_variates, dtype=self.dtype))
            else:
                self._simulation_smoother.set_initial_state_variates(np.array(initial_state_variates, dtype=self.dtype), pretransformed=False)
        else:
            self._simulation_smoother.draw_initial_state_variates(random_state)
        self._simulation_smoother.simulate(simulation_output)