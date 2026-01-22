from __future__ import print_function, absolute_import
import sys
import warnings
from xml.etree.ElementTree import ParseError
from xml.etree.ElementTree import TreeBuilder as _TreeBuilder
from xml.etree.ElementTree import parse as _parse
from xml.etree.ElementTree import tostring
from .common import PY3
from .common import (
def _get_py3_cls():
    """Python 3.3 hides the pure Python code but defusedxml requires it.

    The code is based on test.support.import_fresh_module().
    """
    pymodname = 'xml.etree.ElementTree'
    cmodname = '_elementtree'
    pymod = sys.modules.pop(pymodname, None)
    cmod = sys.modules.pop(cmodname, None)
    sys.modules[cmodname] = None
    try:
        pure_pymod = importlib.import_module(pymodname)
    finally:
        sys.modules[pymodname] = pymod
        if cmod is not None:
            sys.modules[cmodname] = cmod
        else:
            sys.modules.pop(cmodname, None)
        etree_pkg = sys.modules['xml.etree']
        if pymod is not None:
            etree_pkg.ElementTree = pymod
        elif hasattr(etree_pkg, 'ElementTree'):
            del etree_pkg.ElementTree
    _XMLParser = pure_pymod.XMLParser
    _iterparse = pure_pymod.iterparse
    pure_pymod.ParseError = ParseError
    return (_XMLParser, _iterparse)