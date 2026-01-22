import re
import warnings
from numpy import asarray, array, zeros, isscalar, real, imag, vstack
from . import _vode
from . import _dop
from . import _lsoda
class vode(IntegratorBase):
    runner = getattr(_vode, 'dvode', None)
    messages = {-1: 'Excess work done on this call. (Perhaps wrong MF.)', -2: 'Excess accuracy requested. (Tolerances too small.)', -3: 'Illegal input detected. (See printed message.)', -4: 'Repeated error test failures. (Check all input.)', -5: 'Repeated convergence failures. (Perhaps bad Jacobian supplied or wrong choice of MF or tolerances.)', -6: 'Error weight became zero during problem. (Solution component i vanished, and ATOL or ATOL(i) = 0.)'}
    supports_run_relax = 1
    supports_step = 1
    active_global_handle = 0

    def __init__(self, method='adams', with_jacobian=False, rtol=1e-06, atol=1e-12, lband=None, uband=None, order=12, nsteps=500, max_step=0.0, min_step=0.0, first_step=0.0):
        if re.match(method, 'adams', re.I):
            self.meth = 1
        elif re.match(method, 'bdf', re.I):
            self.meth = 2
        else:
            raise ValueError('Unknown integration method %s' % method)
        self.with_jacobian = with_jacobian
        self.rtol = rtol
        self.atol = atol
        self.mu = uband
        self.ml = lband
        self.order = order
        self.nsteps = nsteps
        self.max_step = max_step
        self.min_step = min_step
        self.first_step = first_step
        self.success = 1
        self.initialized = False

    def _determine_mf_and_set_bands(self, has_jac):
        """
        Determine the `MF` parameter (Method Flag) for the Fortran subroutine `dvode`.

        In the Fortran code, the legal values of `MF` are:
            10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25,
            -11, -12, -14, -15, -21, -22, -24, -25
        but this Python wrapper does not use negative values.

        Returns

            mf  = 10*self.meth + miter

        self.meth is the linear multistep method:
            self.meth == 1:  method="adams"
            self.meth == 2:  method="bdf"

        miter is the correction iteration method:
            miter == 0:  Functional iteration; no Jacobian involved.
            miter == 1:  Chord iteration with user-supplied full Jacobian.
            miter == 2:  Chord iteration with internally computed full Jacobian.
            miter == 3:  Chord iteration with internally computed diagonal Jacobian.
            miter == 4:  Chord iteration with user-supplied banded Jacobian.
            miter == 5:  Chord iteration with internally computed banded Jacobian.

        Side effects: If either self.mu or self.ml is not None and the other is None,
        then the one that is None is set to 0.
        """
        jac_is_banded = self.mu is not None or self.ml is not None
        if jac_is_banded:
            if self.mu is None:
                self.mu = 0
            if self.ml is None:
                self.ml = 0
        if has_jac:
            if jac_is_banded:
                miter = 4
            else:
                miter = 1
        elif jac_is_banded:
            if self.ml == self.mu == 0:
                miter = 3
            else:
                miter = 5
        elif self.with_jacobian:
            miter = 2
        else:
            miter = 0
        mf = 10 * self.meth + miter
        return mf

    def reset(self, n, has_jac):
        mf = self._determine_mf_and_set_bands(has_jac)
        if mf == 10:
            lrw = 20 + 16 * n
        elif mf in [11, 12]:
            lrw = 22 + 16 * n + 2 * n * n
        elif mf == 13:
            lrw = 22 + 17 * n
        elif mf in [14, 15]:
            lrw = 22 + 18 * n + (3 * self.ml + 2 * self.mu) * n
        elif mf == 20:
            lrw = 20 + 9 * n
        elif mf in [21, 22]:
            lrw = 22 + 9 * n + 2 * n * n
        elif mf == 23:
            lrw = 22 + 10 * n
        elif mf in [24, 25]:
            lrw = 22 + 11 * n + (3 * self.ml + 2 * self.mu) * n
        else:
            raise ValueError('Unexpected mf=%s' % mf)
        if mf % 10 in [0, 3]:
            liw = 30
        else:
            liw = 30 + n
        rwork = zeros((lrw,), float)
        rwork[4] = self.first_step
        rwork[5] = self.max_step
        rwork[6] = self.min_step
        self.rwork = rwork
        iwork = zeros((liw,), _vode_int_dtype)
        if self.ml is not None:
            iwork[0] = self.ml
        if self.mu is not None:
            iwork[1] = self.mu
        iwork[4] = self.order
        iwork[5] = self.nsteps
        iwork[6] = 2
        self.iwork = iwork
        self.call_args = [self.rtol, self.atol, 1, 1, self.rwork, self.iwork, mf]
        self.success = 1
        self.initialized = False

    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        if self.initialized:
            self.check_handle()
        else:
            self.initialized = True
            self.acquire_new_handle()
        if self.ml is not None and self.ml > 0:
            jac = _vode_banded_jac_wrapper(jac, self.ml, jac_params)
        args = (f, jac, y0, t0, t1) + tuple(self.call_args) + (f_params, jac_params)
        y1, t, istate = self.runner(*args)
        self.istate = istate
        if istate < 0:
            unexpected_istate_msg = f'Unexpected istate={istate:d}'
            warnings.warn('{:s}: {:s}'.format(self.__class__.__name__, self.messages.get(istate, unexpected_istate_msg)), stacklevel=2)
            self.success = 0
        else:
            self.call_args[3] = 2
            self.istate = 2
        return (y1, t)

    def step(self, *args):
        itask = self.call_args[2]
        self.call_args[2] = 2
        r = self.run(*args)
        self.call_args[2] = itask
        return r

    def run_relax(self, *args):
        itask = self.call_args[2]
        self.call_args[2] = 3
        r = self.run(*args)
        self.call_args[2] = itask
        return r