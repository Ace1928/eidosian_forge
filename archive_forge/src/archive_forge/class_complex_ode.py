import re
import warnings
from numpy import asarray, array, zeros, isscalar, real, imag, vstack
from . import _vode
from . import _dop
from . import _lsoda
class complex_ode(ode):
    """
    A wrapper of ode for complex systems.

    This functions similarly as `ode`, but re-maps a complex-valued
    equation system to a real-valued one before using the integrators.

    Parameters
    ----------
    f : callable ``f(t, y, *f_args)``
        Rhs of the equation. t is a scalar, ``y.shape == (n,)``.
        ``f_args`` is set by calling ``set_f_params(*args)``.
    jac : callable ``jac(t, y, *jac_args)``
        Jacobian of the rhs, ``jac[i,j] = d f[i] / d y[j]``.
        ``jac_args`` is set by calling ``set_f_params(*args)``.

    Attributes
    ----------
    t : float
        Current time.
    y : ndarray
        Current variable values.

    Examples
    --------
    For usage examples, see `ode`.

    """

    def __init__(self, f, jac=None):
        self.cf = f
        self.cjac = jac
        if jac is None:
            ode.__init__(self, self._wrap, None)
        else:
            ode.__init__(self, self._wrap, self._wrap_jac)

    def _wrap(self, t, y, *f_args):
        f = self.cf(*(t, y[::2] + 1j * y[1::2]) + f_args)
        self.tmp[::2] = real(f)
        self.tmp[1::2] = imag(f)
        return self.tmp

    def _wrap_jac(self, t, y, *jac_args):
        jac = self.cjac(*(t, y[::2] + 1j * y[1::2]) + jac_args)
        jac_tmp = zeros((2 * jac.shape[0], 2 * jac.shape[1]))
        jac_tmp[1::2, 1::2] = jac_tmp[::2, ::2] = real(jac)
        jac_tmp[1::2, ::2] = imag(jac)
        jac_tmp[::2, 1::2] = -jac_tmp[1::2, ::2]
        ml = getattr(self._integrator, 'ml', None)
        mu = getattr(self._integrator, 'mu', None)
        if ml is not None or mu is not None:
            jac_tmp = _transform_banded_jac(jac_tmp)
        return jac_tmp

    @property
    def y(self):
        return self._y[::2] + 1j * self._y[1::2]

    def set_integrator(self, name, **integrator_params):
        """
        Set integrator by name.

        Parameters
        ----------
        name : str
            Name of the integrator
        **integrator_params
            Additional parameters for the integrator.
        """
        if name == 'zvode':
            raise ValueError('zvode must be used with ode, not complex_ode')
        lband = integrator_params.get('lband')
        uband = integrator_params.get('uband')
        if lband is not None or uband is not None:
            integrator_params['lband'] = 2 * (lband or 0) + 1
            integrator_params['uband'] = 2 * (uband or 0) + 1
        return ode.set_integrator(self, name, **integrator_params)

    def set_initial_value(self, y, t=0.0):
        """Set initial conditions y(t) = y."""
        y = asarray(y)
        self.tmp = zeros(y.size * 2, 'float')
        self.tmp[::2] = real(y)
        self.tmp[1::2] = imag(y)
        return ode.set_initial_value(self, self.tmp, t)

    def integrate(self, t, step=False, relax=False):
        """Find y=y(t), set y as an initial condition, and return y.

        Parameters
        ----------
        t : float
            The endpoint of the integration step.
        step : bool
            If True, and if the integrator supports the step method,
            then perform a single integration step and return.
            This parameter is provided in order to expose internals of
            the implementation, and should not be changed from its default
            value in most cases.
        relax : bool
            If True and if the integrator supports the run_relax method,
            then integrate until t_1 >= t and return. ``relax`` is not
            referenced if ``step=True``.
            This parameter is provided in order to expose internals of
            the implementation, and should not be changed from its default
            value in most cases.

        Returns
        -------
        y : float
            The integrated value at t
        """
        y = ode.integrate(self, t, step, relax)
        return y[::2] + 1j * y[1::2]

    def set_solout(self, solout):
        """
        Set callable to be called at every successful integration step.

        Parameters
        ----------
        solout : callable
            ``solout(t, y)`` is called at each internal integrator step,
            t is a scalar providing the current independent position
            y is the current solution ``y.shape == (n,)``
            solout should return -1 to stop integration
            otherwise it should return None or 0

        """
        if self._integrator.supports_solout:
            self._integrator.set_solout(solout, complex=True)
        else:
            raise TypeError('selected integrator does not support solouta,' + 'choose another one')