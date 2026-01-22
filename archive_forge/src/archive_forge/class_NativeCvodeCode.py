from __future__ import (absolute_import, division, print_function)
import copy
import os
import sys
from ..util import import_
from ._base import _NativeCodeBase, _NativeSysBase, _compile_kwargs
class NativeCvodeCode(_NativeCodeBase):
    wrapper_name = '_cvode_wrapper'
    try:
        _realtype = config['REAL_TYPE']
        _indextype = config['INDEX_TYPE']
    except ModuleNotFoundError:
        _realtype = '#error "realtype_failed-to-import-pycvodes-or-too-old-version"'
        _indextype = '#error "indextype_failed-to-import-pycvodes-or-too-old-version"'
    namespace = {'p_includes': ['"odesys_anyode_iterative.hpp"'], 'p_support_recoverable_error': True, 'p_jacobian_set_to_zero_by_solver': True, 'p_baseclass': 'OdeSysIterativeBase', 'p_realtype': _realtype, 'p_indextype': _indextype}
    _support_roots = True

    def __init__(self, *args, **kwargs):
        self.compile_kwargs = copy.deepcopy(_compile_kwargs)
        self.compile_kwargs['define'] = ['PYCVODES_NO_KLU={}'.format('0' if config.get('KLU', True) else '1'), 'PYCVODES_NO_LAPACK={}'.format('0' if config.get('LAPACK', True) else '1'), 'ANYODE_NO_LAPACK={}'.format('0' if config.get('LAPACK', True) else '1')]
        self.compile_kwargs['include_dirs'].append(get_include())
        self.compile_kwargs['libraries'].extend(_libs.get_libs().split(','))
        self.compile_kwargs['libraries'].extend([l for l in os.environ.get('PYODESYS_LAPACK', 'lapack,blas' if config['LAPACK'] else '').split(',') if l != ''])
        self.compile_kwargs['flags'] = [f for f in os.environ.get('PYODESYS_CVODE_FLAGS', '').split() if f]
        self.compile_kwargs['ldflags'] = [f for f in os.environ.get('PYODESYS_CVODE_LDFLAGS', '').split() if f]
        super(NativeCvodeCode, self).__init__(*args, **kwargs)