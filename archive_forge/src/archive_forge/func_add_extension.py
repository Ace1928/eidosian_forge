import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
from threading import local as tlocal
from functools import reduce
import distutils
from distutils.errors import DistutilsError
def add_extension(self, name, sources, **kw):
    """Add extension to configuration.

        Create and add an Extension instance to the ext_modules list. This
        method also takes the following optional keyword arguments that are
        passed on to the Extension constructor.

        Parameters
        ----------
        name : str
            name of the extension
        sources : seq
            list of the sources. The list of sources may contain functions
            (called source generators) which must take an extension instance
            and a build directory as inputs and return a source file or list of
            source files or None. If None is returned then no sources are
            generated. If the Extension instance has no sources after
            processing all source generators, then no extension module is
            built.
        include_dirs :
        define_macros :
        undef_macros :
        library_dirs :
        libraries :
        runtime_library_dirs :
        extra_objects :
        extra_compile_args :
        extra_link_args :
        extra_f77_compile_args :
        extra_f90_compile_args :
        export_symbols :
        swig_opts :
        depends :
            The depends list contains paths to files or directories that the
            sources of the extension module depend on. If any path in the
            depends list is newer than the extension module, then the module
            will be rebuilt.
        language :
        f2py_options :
        module_dirs :
        extra_info : dict or list
            dict or list of dict of keywords to be appended to keywords.

        Notes
        -----
        The self.paths(...) method is applied to all lists that may contain
        paths.
        """
    ext_args = copy.copy(kw)
    ext_args['name'] = dot_join(self.name, name)
    ext_args['sources'] = sources
    if 'extra_info' in ext_args:
        extra_info = ext_args['extra_info']
        del ext_args['extra_info']
        if isinstance(extra_info, dict):
            extra_info = [extra_info]
        for info in extra_info:
            assert isinstance(info, dict), repr(info)
            dict_append(ext_args, **info)
    self._fix_paths_dict(ext_args)
    libraries = ext_args.get('libraries', [])
    libnames = []
    ext_args['libraries'] = []
    for libname in libraries:
        if isinstance(libname, tuple):
            self._fix_paths_dict(libname[1])
        if '@' in libname:
            lname, lpath = libname.split('@', 1)
            lpath = os.path.abspath(njoin(self.local_path, lpath))
            if os.path.isdir(lpath):
                c = self.get_subpackage(None, lpath, caller_level=2)
                if isinstance(c, Configuration):
                    c = c.todict()
                for l in [l[0] for l in c.get('libraries', [])]:
                    llname = l.split('__OF__', 1)[0]
                    if llname == lname:
                        c.pop('name', None)
                        dict_append(ext_args, **c)
                        break
                continue
        libnames.append(libname)
    ext_args['libraries'] = libnames + ext_args['libraries']
    ext_args['define_macros'] = self.define_macros + ext_args.get('define_macros', [])
    from numpy.distutils.core import Extension
    ext = Extension(**ext_args)
    self.ext_modules.append(ext)
    dist = self.get_distribution()
    if dist is not None:
        self.warn('distutils distribution has been initialized, it may be too late to add an extension ' + name)
    return ext