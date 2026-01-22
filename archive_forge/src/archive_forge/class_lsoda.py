import re
import warnings
from numpy import asarray, array, zeros, isscalar, real, imag, vstack
from . import _vode
from . import _dop
from . import _lsoda
class lsoda(IntegratorBase):
    runner = getattr(_lsoda, 'lsoda', None)
    active_global_handle = 0
    messages = {2: 'Integration successful.', -1: 'Excess work done on this call (perhaps wrong Dfun type).', -2: 'Excess accuracy requested (tolerances too small).', -3: 'Illegal input detected (internal error).', -4: 'Repeated error test failures (internal error).', -5: 'Repeated convergence failures (perhaps bad Jacobian or tolerances).', -6: 'Error weight became zero during problem.', -7: 'Internal workspace insufficient to finish (internal error).'}

    def __init__(self, with_jacobian=False, rtol=1e-06, atol=1e-12, lband=None, uband=None, nsteps=500, max_step=0.0, min_step=0.0, first_step=0.0, ixpr=0, max_hnil=0, max_order_ns=12, max_order_s=5, method=None):
        self.with_jacobian = with_jacobian
        self.rtol = rtol
        self.atol = atol
        self.mu = uband
        self.ml = lband
        self.max_order_ns = max_order_ns
        self.max_order_s = max_order_s
        self.nsteps = nsteps
        self.max_step = max_step
        self.min_step = min_step
        self.first_step = first_step
        self.ixpr = ixpr
        self.max_hnil = max_hnil
        self.success = 1
        self.initialized = False

    def reset(self, n, has_jac):
        if has_jac:
            if self.mu is None and self.ml is None:
                jt = 1
            else:
                if self.mu is None:
                    self.mu = 0
                if self.ml is None:
                    self.ml = 0
                jt = 4
        elif self.mu is None and self.ml is None:
            jt = 2
        else:
            if self.mu is None:
                self.mu = 0
            if self.ml is None:
                self.ml = 0
            jt = 5
        lrn = 20 + (self.max_order_ns + 4) * n
        if jt in [1, 2]:
            lrs = 22 + (self.max_order_s + 4) * n + n * n
        elif jt in [4, 5]:
            lrs = 22 + (self.max_order_s + 5 + 2 * self.ml + self.mu) * n
        else:
            raise ValueError('Unexpected jt=%s' % jt)
        lrw = max(lrn, lrs)
        liw = 20 + n
        rwork = zeros((lrw,), float)
        rwork[4] = self.first_step
        rwork[5] = self.max_step
        rwork[6] = self.min_step
        self.rwork = rwork
        iwork = zeros((liw,), _lsoda_int_dtype)
        if self.ml is not None:
            iwork[0] = self.ml
        if self.mu is not None:
            iwork[1] = self.mu
        iwork[4] = self.ixpr
        iwork[5] = self.nsteps
        iwork[6] = self.max_hnil
        iwork[7] = self.max_order_ns
        iwork[8] = self.max_order_s
        self.iwork = iwork
        self.call_args = [self.rtol, self.atol, 1, 1, self.rwork, self.iwork, jt]
        self.success = 1
        self.initialized = False

    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        if self.initialized:
            self.check_handle()
        else:
            self.initialized = True
            self.acquire_new_handle()
        args = [f, y0, t0, t1] + self.call_args[:-1] + [jac, self.call_args[-1], f_params, 0, jac_params]
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