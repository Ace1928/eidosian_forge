from __future__ import (absolute_import, division, print_function)
import copy
import os
import sys
from ..util import import_
from ._base import _NativeCodeBase, _NativeSysBase, _compile_kwargs
class NativeCvodeSys(_NativeSysBase):
    _NativeCode = NativeCvodeCode
    _native_name = 'cvode'

    def as_standalone(self, out_file=None, compile_kwargs=None):
        from pycompilation.compilation import src2obj, link
        from pycodeexport.util import render_mako_template_to
        compile_kwargs = compile_kwargs or {}
        impl_src = open([f for f in self._native._written_files if f.endswith('.cpp')][0], 'rt').read()
        f = render_mako_template_to(os.path.join(os.path.dirname(__file__), 'sources/standalone_template.cpp'), '%s.cpp' % out_file, {'p_odesys': self, 'p_odesys_impl': impl_src})
        kw = copy.deepcopy(self._native.compile_kwargs)
        kw.update(compile_kwargs)
        objf = src2obj(f, **kw)
        kw['libraries'].append('boost_program_options')
        return link([objf], out_file, **kw)