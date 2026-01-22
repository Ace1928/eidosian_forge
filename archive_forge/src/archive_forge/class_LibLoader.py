imports, including parts of the standard library and installed
import glob
import imp
import os
import sys
from zipimport import zipimporter, ZipImportError
class LibLoader(object):

    def __init__(self):
        self._libs = {}

    def load_module(self, fullname):
        try:
            source_path, so_path, is_package = self._libs[fullname]
        except KeyError:
            raise ValueError('invalid module %s' % fullname)
        _debug("Loading shared library module '%s' from %s", fullname, so_path)
        return load_module(fullname, source_path, so_path=so_path, is_package=is_package)

    def add_lib(self, fullname, path, so_path, is_package):
        self._libs[fullname] = (path, so_path, is_package)

    def knows(self, fullname):
        return fullname in self._libs