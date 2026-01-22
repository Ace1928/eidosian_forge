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
class UfuncifyCodeWrapper(CodeWrapper):
    """Wrapper for Ufuncify"""

    def __init__(self, *args, **kwargs):
        ext_keys = ['include_dirs', 'library_dirs', 'libraries', 'extra_compile_args', 'extra_link_args']
        msg = 'The compilation option kwarg {} is not supported with the numpy backend.'
        for k in ext_keys:
            if k in kwargs.keys():
                warn(msg.format(k))
            kwargs.pop(k, None)
        super().__init__(*args, **kwargs)

    @property
    def command(self):
        command = [sys.executable, 'setup.py', 'build_ext', '--inplace']
        return command

    def wrap_code(self, routines, helpers=None):
        helpers = helpers if helpers is not None else []
        funcname = 'wrapped_' + str(id(routines) + id(helpers))
        workdir = self.filepath or tempfile.mkdtemp('_sympy_compile')
        if not os.access(workdir, os.F_OK):
            os.mkdir(workdir)
        oldwork = os.getcwd()
        os.chdir(workdir)
        try:
            sys.path.append(workdir)
            self._generate_code(routines, helpers)
            self._prepare_files(routines, funcname)
            self._process_files(routines)
            mod = __import__(self.module_name)
        finally:
            sys.path.remove(workdir)
            CodeWrapper._module_counter += 1
            os.chdir(oldwork)
            if not self.filepath:
                try:
                    shutil.rmtree(workdir)
                except OSError:
                    pass
        return self._get_wrapped_function(mod, funcname)

    def _generate_code(self, main_routines, helper_routines):
        all_routines = main_routines + helper_routines
        self.generator.write(all_routines, self.filename, True, self.include_header, self.include_empty)

    def _prepare_files(self, routines, funcname):
        codefilename = self.module_name + '.c'
        with open(codefilename, 'w') as f:
            self.dump_c(routines, f, self.filename, funcname=funcname)
        with open('setup.py', 'w') as f:
            self.dump_setup(f)

    @classmethod
    def _get_wrapped_function(cls, mod, name):
        return getattr(mod, name)

    def dump_setup(self, f):
        setup = _ufunc_setup.substitute(module=self.module_name, filename=self.filename)
        f.write(setup)

    def dump_c(self, routines, f, prefix, funcname=None):
        """Write a C file with Python wrappers

        This file contains all the definitions of the routines in c code.

        Arguments
        ---------
        routines
            List of Routine instances
        f
            File-like object to write the file to
        prefix
            The filename prefix, used to name the imported module.
        funcname
            Name of the main function to be returned.
        """
        if funcname is None:
            if len(routines) == 1:
                funcname = routines[0].name
            else:
                msg = 'funcname must be specified for multiple output routines'
                raise ValueError(msg)
        functions = []
        function_creation = []
        ufunc_init = []
        module = self.module_name
        include_file = '"{}.h"'.format(prefix)
        top = _ufunc_top.substitute(include_file=include_file, module=module)
        name = funcname
        r_index = 0
        py_in, _ = self._partition_args(routines[0].arguments)
        n_in = len(py_in)
        n_out = len(routines)
        form = 'char *{0}{1} = args[{2}];'
        arg_decs = [form.format('in', i, i) for i in range(n_in)]
        arg_decs.extend([form.format('out', i, i + n_in) for i in range(n_out)])
        declare_args = '\n    '.join(arg_decs)
        form = 'npy_intp {0}{1}_step = steps[{2}];'
        step_decs = [form.format('in', i, i) for i in range(n_in)]
        step_decs.extend([form.format('out', i, i + n_in) for i in range(n_out)])
        declare_steps = '\n    '.join(step_decs)
        form = '*(double *)in{0}'
        call_args = ', '.join([form.format(a) for a in range(n_in)])
        form = '{0}{1} += {0}{1}_step;'
        step_incs = [form.format('in', i) for i in range(n_in)]
        step_incs.extend([form.format('out', i, i) for i in range(n_out)])
        step_increments = '\n        '.join(step_incs)
        n_types = n_in + n_out
        types = '{' + ', '.join(['NPY_DOUBLE'] * n_types) + '};'
        docstring = '"Created in SymPy with Ufuncify"'
        function_creation.append('PyObject *ufunc{};'.format(r_index))
        init_form = _ufunc_init_form.substitute(module=module, funcname=name, docstring=docstring, n_in=n_in, n_out=n_out, ind=r_index)
        ufunc_init.append(init_form)
        outcalls = [_ufunc_outcalls.substitute(outnum=i, call_args=call_args, funcname=routines[i].name) for i in range(n_out)]
        body = _ufunc_body.substitute(module=module, funcname=name, declare_args=declare_args, declare_steps=declare_steps, call_args=call_args, step_increments=step_increments, n_types=n_types, types=types, outcalls='\n        '.join(outcalls))
        functions.append(body)
        body = '\n\n'.join(functions)
        ufunc_init = '\n    '.join(ufunc_init)
        function_creation = '\n    '.join(function_creation)
        bottom = _ufunc_bottom.substitute(module=module, ufunc_init=ufunc_init, function_creation=function_creation)
        text = [top, body, bottom]
        f.write('\n\n'.join(text))

    def _partition_args(self, args):
        """Group function arguments into categories."""
        py_in = []
        py_out = []
        for arg in args:
            if isinstance(arg, OutputArgument):
                py_out.append(arg)
            elif isinstance(arg, InOutArgument):
                raise ValueError("Ufuncify doesn't support InOutArguments")
            else:
                py_in.append(arg)
        return (py_in, py_out)