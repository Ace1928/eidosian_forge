from __future__ import absolute_import, print_function
import cython
from .. import __version__
import collections
import contextlib
import hashlib
import os
import shutil
import subprocess
import re, sys, time
from glob import iglob
from io import open as io_open
from os.path import relpath as _relpath
import zipfile
from .. import Utils
from ..Utils import (cached_function, cached_method, path_exists,
from ..Compiler import Errors
from ..Compiler.Main import Context
from ..Compiler.Options import (CompilationOptions, default_options,
class DependencyTree(object):

    def __init__(self, context, quiet=False):
        self.context = context
        self.quiet = quiet
        self._transitive_cache = {}

    def parse_dependencies(self, source_filename):
        if path_exists(source_filename):
            source_filename = os.path.normpath(source_filename)
        return parse_dependencies(source_filename)

    @cached_method
    def included_files(self, filename):
        all = set()
        for include in self.parse_dependencies(filename)[1]:
            include_path = join_path(os.path.dirname(filename), include)
            if not path_exists(include_path):
                include_path = self.context.find_include_file(include, source_file_path=filename)
            if include_path:
                if '.' + os.path.sep in include_path:
                    include_path = os.path.normpath(include_path)
                all.add(include_path)
                all.update(self.included_files(include_path))
            elif not self.quiet:
                print(u"Unable to locate '%s' referenced from '%s'" % (filename, include))
        return all

    @cached_method
    def cimports_externs_incdirs(self, filename):
        cimports, includes, externs = self.parse_dependencies(filename)[:3]
        cimports = set(cimports)
        externs = set(externs)
        incdirs = set()
        for include in self.included_files(filename):
            included_cimports, included_externs, included_incdirs = self.cimports_externs_incdirs(include)
            cimports.update(included_cimports)
            externs.update(included_externs)
            incdirs.update(included_incdirs)
        externs, incdir = normalize_existing(filename, externs)
        if incdir:
            incdirs.add(incdir)
        return (tuple(cimports), externs, incdirs)

    def cimports(self, filename):
        return self.cimports_externs_incdirs(filename)[0]

    def package(self, filename):
        return package(filename)

    def fully_qualified_name(self, filename):
        return fully_qualified_name(filename)

    @cached_method
    def find_pxd(self, module, filename=None):
        is_relative = module[0] == '.'
        if is_relative and (not filename):
            raise NotImplementedError('New relative imports.')
        if filename is not None:
            module_path = module.split('.')
            if is_relative:
                module_path.pop(0)
            package_path = list(self.package(filename))
            while module_path and (not module_path[0]):
                try:
                    package_path.pop()
                except IndexError:
                    return None
                module_path.pop(0)
            relative = '.'.join(package_path + module_path)
            pxd = self.context.find_pxd_file(relative, source_file_path=filename)
            if pxd:
                return pxd
        if is_relative:
            return None
        return self.context.find_pxd_file(module, source_file_path=filename)

    @cached_method
    def cimported_files(self, filename):
        filename_root, filename_ext = os.path.splitext(filename)
        if filename_ext in ('.pyx', '.py') and path_exists(filename_root + '.pxd'):
            pxd_list = [filename_root + '.pxd']
        else:
            pxd_list = []
        for module in self.cimports(filename):
            if module[:7] == 'cython.' or module == 'cython':
                continue
            pxd_file = self.find_pxd(module, filename)
            if pxd_file is not None:
                pxd_list.append(pxd_file)
        return tuple(pxd_list)

    @cached_method
    def immediate_dependencies(self, filename):
        all_deps = {filename}
        all_deps.update(self.cimported_files(filename))
        all_deps.update(self.included_files(filename))
        return all_deps

    def all_dependencies(self, filename):
        return self.transitive_merge(filename, self.immediate_dependencies, set.union)

    @cached_method
    def timestamp(self, filename):
        return os.path.getmtime(filename)

    def extract_timestamp(self, filename):
        return (self.timestamp(filename), filename)

    def newest_dependency(self, filename):
        return max([self.extract_timestamp(f) for f in self.all_dependencies(filename)])

    def transitive_fingerprint(self, filename, module, compilation_options):
        """
        Return a fingerprint of a cython file that is about to be cythonized.

        Fingerprints are looked up in future compilations. If the fingerprint
        is found, the cythonization can be skipped. The fingerprint must
        incorporate everything that has an influence on the generated code.
        """
        try:
            m = hashlib.sha1(__version__.encode('UTF-8'))
            m.update(file_hash(filename).encode('UTF-8'))
            for x in sorted(self.all_dependencies(filename)):
                if os.path.splitext(x)[1] not in ('.c', '.cpp', '.h'):
                    m.update(file_hash(x).encode('UTF-8'))
            m.update(str((module.language, getattr(module, 'py_limited_api', False), getattr(module, 'np_pythran', False))).encode('UTF-8'))
            m.update(compilation_options.get_fingerprint().encode('UTF-8'))
            return m.hexdigest()
        except IOError:
            return None

    def distutils_info0(self, filename):
        info = self.parse_dependencies(filename)[3]
        kwds = info.values
        cimports, externs, incdirs = self.cimports_externs_incdirs(filename)
        basedir = os.getcwd()
        if externs:
            externs = _make_relative(externs, basedir)
            if 'depends' in kwds:
                kwds['depends'] = list(set(kwds['depends']).union(externs))
            else:
                kwds['depends'] = list(externs)
        if incdirs:
            include_dirs = list(kwds.get('include_dirs', []))
            for inc in _make_relative(incdirs, basedir):
                if inc not in include_dirs:
                    include_dirs.append(inc)
            kwds['include_dirs'] = include_dirs
        return info

    def distutils_info(self, filename, aliases=None, base=None):
        return self.transitive_merge(filename, self.distutils_info0, DistutilsInfo.merge).subs(aliases).merge(base)

    def transitive_merge(self, node, extract, merge):
        try:
            seen = self._transitive_cache[extract, merge]
        except KeyError:
            seen = self._transitive_cache[extract, merge] = {}
        return self.transitive_merge_helper(node, extract, merge, seen, {}, self.cimported_files)[0]

    def transitive_merge_helper(self, node, extract, merge, seen, stack, outgoing):
        if node in seen:
            return (seen[node], None)
        deps = extract(node)
        if node in stack:
            return (deps, node)
        try:
            stack[node] = len(stack)
            loop = None
            for next in outgoing(node):
                sub_deps, sub_loop = self.transitive_merge_helper(next, extract, merge, seen, stack, outgoing)
                if sub_loop is not None:
                    if loop is not None and stack[loop] < stack[sub_loop]:
                        pass
                    else:
                        loop = sub_loop
                deps = merge(deps, sub_deps)
            if loop == node:
                loop = None
            if loop is None:
                seen[node] = deps
            return (deps, loop)
        finally:
            del stack[node]