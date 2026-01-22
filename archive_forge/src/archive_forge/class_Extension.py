import re
from distutils.extension import Extension as old_Extension
class Extension(old_Extension):
    """
    Parameters
    ----------
    name : str
        Extension name.
    sources : list of str
        List of source file locations relative to the top directory of
        the package.
    extra_compile_args : list of str
        Extra command line arguments to pass to the compiler.
    extra_f77_compile_args : list of str
        Extra command line arguments to pass to the fortran77 compiler.
    extra_f90_compile_args : list of str
        Extra command line arguments to pass to the fortran90 compiler.
    """

    def __init__(self, name, sources, include_dirs=None, define_macros=None, undef_macros=None, library_dirs=None, libraries=None, runtime_library_dirs=None, extra_objects=None, extra_compile_args=None, extra_link_args=None, export_symbols=None, swig_opts=None, depends=None, language=None, f2py_options=None, module_dirs=None, extra_c_compile_args=None, extra_cxx_compile_args=None, extra_f77_compile_args=None, extra_f90_compile_args=None):
        old_Extension.__init__(self, name, [], include_dirs=include_dirs, define_macros=define_macros, undef_macros=undef_macros, library_dirs=library_dirs, libraries=libraries, runtime_library_dirs=runtime_library_dirs, extra_objects=extra_objects, extra_compile_args=extra_compile_args, extra_link_args=extra_link_args, export_symbols=export_symbols)
        self.sources = sources
        self.swig_opts = swig_opts or []
        if isinstance(self.swig_opts, str):
            import warnings
            msg = 'swig_opts is specified as a string instead of a list'
            warnings.warn(msg, SyntaxWarning, stacklevel=2)
            self.swig_opts = self.swig_opts.split()
        self.depends = depends or []
        self.language = language
        self.f2py_options = f2py_options or []
        self.module_dirs = module_dirs or []
        self.extra_c_compile_args = extra_c_compile_args or []
        self.extra_cxx_compile_args = extra_cxx_compile_args or []
        self.extra_f77_compile_args = extra_f77_compile_args or []
        self.extra_f90_compile_args = extra_f90_compile_args or []
        return

    def has_cxx_sources(self):
        for source in self.sources:
            if cxx_ext_re(str(source)):
                return True
        return False

    def has_f2py_sources(self):
        for source in self.sources:
            if fortran_pyf_ext_re(source):
                return True
        return False