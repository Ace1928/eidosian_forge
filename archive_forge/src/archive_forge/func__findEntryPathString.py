from __future__ import annotations
import inspect
import sys
import warnings
import zipimport
from os.path import dirname, split as splitpath
from zope.interface import Interface, implementer
from twisted.python.compat import nativeString
from twisted.python.components import registerAdapter
from twisted.python.filepath import FilePath, UnlistableError
from twisted.python.reflect import namedAny
from twisted.python.zippath import ZipArchive
def _findEntryPathString(self, modobj):
    """
        Determine where a given Python module object came from by looking at path
        entries.
        """
    topPackageObj = modobj
    while '.' in topPackageObj.__name__:
        topPackageObj = self.moduleDict['.'.join(topPackageObj.__name__.split('.')[:-1])]
    if _isPackagePath(FilePath(topPackageObj.__file__)):
        rval = dirname(dirname(topPackageObj.__file__))
    else:
        rval = dirname(topPackageObj.__file__)
    if rval not in self.importerCache:
        warnings.warn('%s (for module %s) not in path importer cache (PEP 302 violation - check your local configuration).' % (rval, modobj.__name__), stacklevel=3)
    return rval