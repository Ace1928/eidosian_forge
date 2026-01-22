import sys
import os
import shutil
import tempfile
from subprocess import STDOUT, CalledProcessError, check_output
from string import Template
from warnings import warn
from sympy.core.cache import cacheit
from sympy.core.function import Lambda
from sympy.core.relational import Eq
from sympy.core.symbol import Dummy, Symbol
from sympy.tensor.indexed import Idx, IndexedBase
from sympy.utilities.codegen import (make_routine, get_code_generator,
from sympy.utilities.iterables import iterable
from sympy.utilities.lambdify import implemented_function
from sympy.utilities.decorator import doctest_depends_on
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
from setuptools.extension import Extension
from setuptools import setup
from numpy import get_include
class CythonCodeWrapper(CodeWrapper):
    """Wrapper that uses Cython"""
    setup_template = 'from setuptools import setup\nfrom setuptools import Extension\nfrom Cython.Build import cythonize\ncy_opts = {cythonize_options}\n{np_import}\next_mods = [Extension(\n    {ext_args},\n    include_dirs={include_dirs},\n    library_dirs={library_dirs},\n    libraries={libraries},\n    extra_compile_args={extra_compile_args},\n    extra_link_args={extra_link_args}\n)]\nsetup(ext_modules=cythonize(ext_mods, **cy_opts))\n'
    _cythonize_options = {'compiler_directives': {'language_level': '3'}}
    pyx_imports = 'import numpy as np\ncimport numpy as np\n\n'
    pyx_header = "cdef extern from '{header_file}.h':\n    {prototype}\n\n"
    pyx_func = 'def {name}_c({arg_string}):\n\n{declarations}{body}'
    std_compile_flag = '-std=c99'

    def __init__(self, *args, **kwargs):
        """Instantiates a Cython code wrapper.

        The following optional parameters get passed to ``setuptools.Extension``
        for building the Python extension module. Read its documentation to
        learn more.

        Parameters
        ==========
        include_dirs : [list of strings]
            A list of directories to search for C/C++ header files (in Unix
            form for portability).
        library_dirs : [list of strings]
            A list of directories to search for C/C++ libraries at link time.
        libraries : [list of strings]
            A list of library names (not filenames or paths) to link against.
        extra_compile_args : [list of strings]
            Any extra platform- and compiler-specific information to use when
            compiling the source files in 'sources'.  For platforms and
            compilers where "command line" makes sense, this is typically a
            list of command-line arguments, but for other platforms it could be
            anything. Note that the attribute ``std_compile_flag`` will be
            appended to this list.
        extra_link_args : [list of strings]
            Any extra platform- and compiler-specific information to use when
            linking object files together to create the extension (or to create
            a new static Python interpreter). Similar interpretation as for
            'extra_compile_args'.
        cythonize_options : [dictionary]
            Keyword arguments passed on to cythonize.

        """
        self._include_dirs = kwargs.pop('include_dirs', [])
        self._library_dirs = kwargs.pop('library_dirs', [])
        self._libraries = kwargs.pop('libraries', [])
        self._extra_compile_args = kwargs.pop('extra_compile_args', [])
        self._extra_compile_args.append(self.std_compile_flag)
        self._extra_link_args = kwargs.pop('extra_link_args', [])
        self._cythonize_options = kwargs.pop('cythonize_options', self._cythonize_options)
        self._need_numpy = False
        super().__init__(*args, **kwargs)

    @property
    def command(self):
        command = [sys.executable, 'setup.py', 'build_ext', '--inplace']
        return command

    def _prepare_files(self, routine, build_dir=os.curdir):
        pyxfilename = self.module_name + '.pyx'
        codefilename = '%s.%s' % (self.filename, self.generator.code_extension)
        with open(os.path.join(build_dir, pyxfilename), 'w') as f:
            self.dump_pyx([routine], f, self.filename)
        ext_args = [repr(self.module_name), repr([pyxfilename, codefilename])]
        if self._need_numpy:
            np_import = 'import numpy as np\n'
            self._include_dirs.append('np.get_include()')
        else:
            np_import = ''
        with open(os.path.join(build_dir, 'setup.py'), 'w') as f:
            includes = str(self._include_dirs).replace("'np.get_include()'", 'np.get_include()')
            f.write(self.setup_template.format(ext_args=', '.join(ext_args), np_import=np_import, include_dirs=includes, library_dirs=self._library_dirs, libraries=self._libraries, extra_compile_args=self._extra_compile_args, extra_link_args=self._extra_link_args, cythonize_options=self._cythonize_options))

    @classmethod
    def _get_wrapped_function(cls, mod, name):
        return getattr(mod, name + '_c')

    def dump_pyx(self, routines, f, prefix):
        """Write a Cython file with Python wrappers

        This file contains all the definitions of the routines in c code and
        refers to the header file.

        Arguments
        ---------
        routines
            List of Routine instances
        f
            File-like object to write the file to
        prefix
            The filename prefix, used to refer to the proper header file.
            Only the basename of the prefix is used.
        """
        headers = []
        functions = []
        for routine in routines:
            prototype = self.generator.get_prototype(routine)
            headers.append(self.pyx_header.format(header_file=prefix, prototype=prototype))
            py_rets, py_args, py_loc, py_inf = self._partition_args(routine.arguments)
            name = routine.name
            arg_string = ', '.join((self._prototype_arg(arg) for arg in py_args))
            local_decs = []
            for arg, val in py_inf.items():
                proto = self._prototype_arg(arg)
                mat, ind = [self._string_var(v) for v in val]
                local_decs.append('    cdef {} = {}.shape[{}]'.format(proto, mat, ind))
            local_decs.extend(['    cdef {}'.format(self._declare_arg(a)) for a in py_loc])
            declarations = '\n'.join(local_decs)
            if declarations:
                declarations = declarations + '\n'
            args_c = ', '.join([self._call_arg(a) for a in routine.arguments])
            rets = ', '.join([self._string_var(r.name) for r in py_rets])
            if routine.results:
                body = '    return %s(%s)' % (routine.name, args_c)
                if rets:
                    body = body + ', ' + rets
            else:
                body = '    %s(%s)\n' % (routine.name, args_c)
                body = body + '    return ' + rets
            functions.append(self.pyx_func.format(name=name, arg_string=arg_string, declarations=declarations, body=body))
        if self._need_numpy:
            f.write(self.pyx_imports)
        f.write('\n'.join(headers))
        f.write('\n'.join(functions))

    def _partition_args(self, args):
        """Group function arguments into categories."""
        py_args = []
        py_returns = []
        py_locals = []
        py_inferred = {}
        for arg in args:
            if isinstance(arg, OutputArgument):
                py_returns.append(arg)
                py_locals.append(arg)
            elif isinstance(arg, InOutArgument):
                py_returns.append(arg)
                py_args.append(arg)
            else:
                py_args.append(arg)
            if isinstance(arg, (InputArgument, InOutArgument)) and arg.dimensions:
                dims = [d[1] + 1 for d in arg.dimensions]
                sym_dims = [(i, d) for i, d in enumerate(dims) if isinstance(d, Symbol)]
                for i, d in sym_dims:
                    py_inferred[d] = (arg.name, i)
        for arg in args:
            if arg.name in py_inferred:
                py_inferred[arg] = py_inferred.pop(arg.name)
        py_args = [a for a in py_args if a not in py_inferred]
        return (py_returns, py_args, py_locals, py_inferred)

    def _prototype_arg(self, arg):
        mat_dec = 'np.ndarray[{mtype}, ndim={ndim}] {name}'
        np_types = {'double': 'np.double_t', 'int': 'np.int_t'}
        t = arg.get_datatype('c')
        if arg.dimensions:
            self._need_numpy = True
            ndim = len(arg.dimensions)
            mtype = np_types[t]
            return mat_dec.format(mtype=mtype, ndim=ndim, name=self._string_var(arg.name))
        else:
            return '%s %s' % (t, self._string_var(arg.name))

    def _declare_arg(self, arg):
        proto = self._prototype_arg(arg)
        if arg.dimensions:
            shape = '(' + ','.join((self._string_var(i[1] + 1) for i in arg.dimensions)) + ')'
            return proto + ' = np.empty({shape})'.format(shape=shape)
        else:
            return proto + ' = 0'

    def _call_arg(self, arg):
        if arg.dimensions:
            t = arg.get_datatype('c')
            return '<{}*> {}.data'.format(t, self._string_var(arg.name))
        elif isinstance(arg, ResultBase):
            return '&{}'.format(self._string_var(arg.name))
        else:
            return self._string_var(arg.name)

    def _string_var(self, var):
        printer = self.generator.printer.doprint
        return printer(var)