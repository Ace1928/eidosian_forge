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
class _NativeCodeBase(Cpp_Code):
    """ Base class for generated code.

    Note kwargs ``namespace_override`` which allows the user to customize
    the variables used when rendering the template.
    """
    wrapper_name = None
    basedir = os.path.dirname(__file__)
    templates = ('sources/odesys_anyode_template.cpp',)
    _written_files = ()
    build_files = ()
    source_files = ('odesys_anyode.cpp',)
    obj_files = ('odesys_anyode.o',)
    _save_temp = False
    namespace_default = {'p_anon': None}
    namespace = {'p_includes': ['"odesys_anyode.hpp"'], 'p_support_recoverable_error': False, 'p_jacobian_set_to_zero_by_solver': False, 'p_realtype': 'double', 'p_indextype': 'int', 'p_baseclass': 'OdeSysBase'}
    _support_roots = False

    def __init__(self, odesys, *args, **kwargs):
        if Cpp_Code is object:
            raise ModuleNotFoundError('failed to import Cpp_Code from pycodeexport')
        if compile_sources is None:
            raise ModuleNotFoundError('failed to import compile_sources from pycompilation')
        if odesys.nroots > 0 and (not self._support_roots):
            raise ValueError('%s does not support nroots > 0' % self.__class__.__name__)
        self.namespace_override = kwargs.pop('namespace_override', {})
        self.namespace_extend = kwargs.pop('namespace_extend', {})
        self.tempdir_basename = '_pycodeexport_pyodesys_%s' % self.__class__.__name__
        self.obj_files = self.obj_files + ('%s%s' % (self.wrapper_name, _obj_suffix),)
        self.so_file = '%s%s' % (self.wrapper_name, '.so')
        _wrapper_src = pkg_resources.resource_filename(__name__, 'sources/%s.pyx' % self.wrapper_name)
        if cachedir is None:
            raise ImportError("No module named appdirs (needed for caching). Install 'appdirs' using e.g. pip/conda.")
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
        _wrapper_src = os.path.join(cachedir, '%s%s' % (self.wrapper_name, '.pyx'))
        shutil.copy(pkg_resources.resource_filename(__name__, 'sources/%s.pyx' % self.wrapper_name), _wrapper_src)
        _wrapper_obj = os.path.join(cachedir, '%s%s' % (self.wrapper_name, _obj_suffix))
        prebuild = {_wrapper_src: _wrapper_obj}
        self.build_files = self.build_files + tuple(prebuild.values())
        self.odesys = odesys
        for _src, _dest in prebuild.items():
            if not os.path.exists(_dest):
                tmpdir = tempfile.mkdtemp()
                try:
                    compile_sources([_src], cwd=tmpdir, metadir=cachedir, logger=logger, **self.compile_kwargs)
                    shutil.copy(os.path.join(tmpdir, os.path.basename(_src)[:-4] + '.o'), _dest)
                finally:
                    if not kwargs.get('save_temp', False):
                        shutil.rmtree(tmpdir)
                if not os.path.exists(_dest):
                    raise OSError('Failed to place prebuilt file at: %s' % _dest)
        super(_NativeCodeBase, self).__init__(*args, logger=logger, **kwargs)

    def variables(self):
        ny = self.odesys.ny
        if self.odesys.band is not None:
            raise NotImplementedError('Banded jacobian not yet implemented.')
        all_invar = tuple(self.odesys.all_invariants())
        ninvar = len(all_invar)
        jac = self.odesys.get_jac()
        nnz = self.odesys.nnz
        all_exprs = self.odesys.exprs + all_invar
        if jac is not False and nnz < 0:
            jac_dfdx = list(reduce(add, jac.tolist() + self.odesys.get_dfdx().tolist()))
            all_exprs += tuple(jac_dfdx)
            nj = len(jac_dfdx)
        elif jac is not False and nnz >= 0:
            jac_dfdx = list(reduce(add, jac.tolist()))
            all_exprs += tuple(jac_dfdx)
            nj = len(jac_dfdx)
        else:
            nj = 0
        jtimes = self.odesys.get_jtimes()
        if jtimes is not False:
            v, jtimes_exprs = jtimes
            all_exprs += tuple(jtimes_exprs)
            njtimes = len(jtimes_exprs)
        else:
            v = ()
            jtimes_exprs = ()
            njtimes = 0
        subsd = {k: self.odesys.be.Symbol('y[%d]' % idx) for idx, k in enumerate(self.odesys.dep)}
        subsd[self.odesys.indep] = self.odesys.be.Symbol('x')
        if jtimes is not False:
            subsd.update({k: self.odesys.be.Symbol('v[%d]' % idx) for idx, k in enumerate(v)})
        subsd.update({k: self.odesys.be.Symbol('m_p[%d]' % idx) for idx, k in enumerate(self.odesys.params)})

        def common_cse_symbols():
            idx = 0
            while True:
                yield self.odesys.be.Symbol('m_p_cse[%d]' % idx)
                idx += 1

        def _ccode(expr):
            return self.odesys.be.ccode(expr.xreplace(subsd))
        if os.getenv('PYODESYS_NATIVE_CSE', '1') == '1':
            cse_cb = self.odesys.be.cse
        else:
            logger.info('Not using common subexpression elimination (disabled by PYODESYS_NATIVE_CSE)')
            cse_cb = lambda exprs, **kwargs: ([], exprs)
        common_cses, common_exprs = cse_cb(all_exprs, symbols=self.odesys.be.numbered_symbols('cse_temporary'), ignore=(self.odesys.indep,) + self.odesys.dep + v)
        common_cse_subs = {}
        comm_cse_symbs = common_cse_symbols()
        for symb, subexpr in common_cses:
            for expr in common_exprs:
                if symb in expr.free_symbols:
                    common_cse_subs[symb] = next(comm_cse_symbs)
                    break
        common_cses = [(x.xreplace(common_cse_subs), expr.xreplace(common_cse_subs)) for x, expr in common_cses]
        common_exprs = [expr.xreplace(common_cse_subs) for expr in common_exprs]
        rhs_cses, rhs_exprs = cse_cb(common_exprs[:ny], symbols=self.odesys.be.numbered_symbols('cse'))
        if all_invar:
            invar_cses, invar_exprs = cse_cb(common_exprs[ny:ny + ninvar], symbols=self.odesys.be.numbered_symbols('cse'))
        if jac is not False:
            jac_cses, jac_exprs = cse_cb(common_exprs[ny + ninvar:ny + ninvar + nj], symbols=self.odesys.be.numbered_symbols('cse'))
        if jtimes is not False:
            jtimes_cses, jtimes_exprs = cse_cb(common_exprs[ny + ninvar + nj:ny + ninvar + nj + njtimes], symbols=self.odesys.be.numbered_symbols('cse'))
        first_step = self.odesys.first_step_expr
        if first_step is not None:
            first_step_cses, first_step_exprs = cse_cb([first_step], symbols=self.odesys.be.numbered_symbols('cse'))
        if self.odesys.roots is not None:
            roots_cses, roots_exprs = cse_cb(self.odesys.roots, symbols=self.odesys.be.numbered_symbols('cse'))
        ns = dict(_message_for_rendered=['-*- mode: read-only -*-', 'This file was generated using pyodesys-%s at %s' % (__version__, dt.now().isoformat())], p_odesys=self.odesys, p_common={'cses': [(symb.name, _ccode(expr)) for symb, expr in common_cses], 'nsubs': len(common_cse_subs)}, p_rhs={'cses': [(symb.name, _ccode(expr)) for symb, expr in rhs_cses], 'exprs': list(map(_ccode, rhs_exprs))}, p_jtimes=None if jtimes is False else {'cses': [(symb.name, _ccode(expr)) for symb, expr in jtimes_cses], 'exprs': list(map(_ccode, jtimes_exprs))}, p_jac_dense=None if jac is False or nnz >= 0 else {'cses': [(symb.name, _ccode(expr)) for symb, expr in jac_cses], 'exprs': {(idx // ny, idx % ny): _ccode(expr) for idx, expr in enumerate(jac_exprs[:ny * ny])}, 'dfdt_exprs': list(map(_ccode, jac_exprs[ny * ny:]))}, p_jac_sparse=None if jac is False or nnz < 0 else {'cses': [(symb.name, _ccode(expr)) for symb, expr in jac_cses], 'exprs': list(map(_ccode, jac_exprs[:nj])), 'colptrs': self.odesys._colptrs, 'rowvals': self.odesys._rowvals}, p_first_step=None if first_step is None else {'cses': first_step_cses, 'expr': _ccode(first_step_exprs[0])}, p_roots=None if self.odesys.roots is None else {'cses': [(symb.name, _ccode(expr)) for symb, expr in roots_cses], 'exprs': list(map(_ccode, roots_exprs))}, p_invariants=None if all_invar == () else {'cses': [(symb.name, _ccode(expr)) for symb, expr in invar_cses], 'exprs': list(map(_ccode, invar_exprs))}, p_nroots=self.odesys.nroots, p_constructor=[], p_get_dx_max=False)
        ns.update(self.namespace_default)
        ns.update(self.namespace)
        ns.update(self.namespace_override)
        for k, v in self.namespace_extend.items():
            ns[k].extend(v)
        return ns