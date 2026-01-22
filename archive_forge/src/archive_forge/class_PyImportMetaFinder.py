imports, including parts of the standard library and installed
import glob
import importlib
import os
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ExtensionFileLoader, SourceFileLoader
from importlib.util import spec_from_file_location
class PyImportMetaFinder(MetaPathFinder):

    def __init__(self, extension=PY_EXT, pyxbuild_dir=None, inplace=False, language_level=None):
        self.pyxbuild_dir = pyxbuild_dir
        self.inplace = inplace
        self.language_level = language_level
        self.extension = extension
        self.uncompilable_modules = {}
        self.blocked_modules = ['Cython', 'pyxbuild', 'pyximport.pyxbuild', 'distutils', 'cython']
        self.blocked_packages = ['Cython.', 'distutils.']

    def find_spec(self, fullname, path, target=None):
        if fullname in sys.modules:
            return None
        if any([fullname.startswith(pkg) for pkg in self.blocked_packages]):
            return None
        if fullname in self.blocked_modules:
            return None
        self.blocked_modules.append(fullname)
        name = fullname
        if not path:
            path = [os.getcwd()]
        try:
            for entry in path:
                if os.path.isdir(os.path.join(entry, name)):
                    filename = os.path.join(entry, name, '__init__' + self.extension)
                    submodule_locations = [os.path.join(entry, name)]
                else:
                    filename = os.path.join(entry, name + self.extension)
                    submodule_locations = None
                if not os.path.exists(filename):
                    continue
                return spec_from_file_location(fullname, filename, loader=PyxImportLoader(filename, self.pyxbuild_dir, self.inplace, self.language_level), submodule_search_locations=submodule_locations)
        finally:
            self.blocked_modules.pop()
        return None