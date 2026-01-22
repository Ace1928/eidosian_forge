from __future__ import (absolute_import, division, print_function)
from datetime import datetime as dt
from functools import reduce
import logging
from operator import add
import os
import shutil
import sys
import tempfile
import numpy as np
import pkg_resources
from ..symbolic import SymbolicSys
from .. import __version__
class _NativeSysBase(SymbolicSys):
    _NativeCode = None
    _native_name = None

    def __init__(self, *args, **kwargs):
        namespace_override = kwargs.pop('namespace_override', {})
        namespace_extend = kwargs.pop('namespace_extend', {})
        save_temp = kwargs.pop('save_temp', False)
        if 'init_indep' not in kwargs:
            kwargs['init_indep'] = True
            kwargs['init_dep'] = True
        super(_NativeSysBase, self).__init__(*args, **kwargs)
        self._native = self._NativeCode(self, save_temp=save_temp, namespace_override=namespace_override, namespace_extend=namespace_extend)

    def integrate(self, *args, **kwargs):
        integrator = kwargs.pop('integrator', 'native')
        if integrator not in ('native', self._native_name):
            raise ValueError('Got incompatible kwargs integrator=%s' % integrator)
        else:
            kwargs['integrator'] = 'native'
        return super(_NativeSysBase, self).integrate(*args, **kwargs)

    def _integrate_native(self, intern_x, intern_y0, intern_p, force_predefined=False, atol=1e-08, rtol=1e-08, nsteps=500, first_step=0.0, **kwargs):
        atol = np.atleast_1d(atol)
        y0 = np.ascontiguousarray(intern_y0, dtype=np.float64)
        params = np.ascontiguousarray(intern_p, dtype=np.float64)
        if atol.size != 1 and atol.size != self.ny:
            raise ValueError('atol needs to be of length 1 or %d' % self.ny)
        if intern_x.shape[-1] == 2 and (not force_predefined):
            intern_xout, yout, info = self._native.mod.integrate_adaptive(y0=y0, x0=np.ascontiguousarray(intern_x[:, 0], dtype=np.float64), xend=np.ascontiguousarray(intern_x[:, 1], dtype=np.float64), params=params, atol=atol, rtol=rtol, mxsteps=nsteps, dx0=first_step, **kwargs)
        else:
            yout, info = self._native.mod.integrate_predefined(y0=y0, xout=np.ascontiguousarray(intern_x, dtype=np.float64), params=params, atol=atol, rtol=rtol, mxsteps=nsteps, dx0=first_step, **kwargs)
            intern_xout = intern_x
        for idx in range(len(info)):
            info[idx]['internal_xout'] = intern_xout[idx]
            info[idx]['internal_yout'] = yout[idx]
            info[idx]['internal_params'] = intern_p[idx, ...]
            if 'nfev' not in info[idx] and 'n_rhs_evals' in info[idx]:
                info[idx]['nfev'] = info[idx]['n_rhs_evals']
            if 'njev' not in info[idx] and 'dense_n_dls_jac_evals' in info[idx]:
                info[idx]['njev'] = info[idx]['dense_n_dls_jac_evals']
            if 'njvev' not in info[idx] and 'krylov_n_jac_times_evals' in info[idx]:
                info[idx]['njvev'] = info[idx]['krylov_n_jac_times_evals']
        return info