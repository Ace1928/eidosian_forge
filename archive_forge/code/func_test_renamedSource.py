from __future__ import annotations
import sys
import warnings
from io import StringIO
from typing import Mapping, Sequence, TypeVar
from unittest import TestResult
from twisted.python.filepath import FilePath
from twisted.trial._synctest import (
from twisted.trial.unittest import SynchronousTestCase
import warnings
import warnings
def test_renamedSource(self) -> None:
    """
        Warnings emitted by a function defined in a file which has been renamed
        since it was initially compiled can still be flushed.

        This is testing the code which specifically supports working around the
        unfortunate behavior of CPython to write a .py source file name into
        the .pyc files it generates and then trust that it is correct in
        various places.  If source files are renamed, .pyc files may not be
        regenerated, but they will contain incorrect filenames.
        """
    package = FilePath(self.mktemp().encode('utf-8')).child(b'twisted_private_helper')
    package.makedirs()
    package.child(b'__init__.py').setContent(b'')
    package.child(b'module.py').setContent(b'\nimport warnings\ndef foo():\n    warnings.warn("oh no")\n')
    pathEntry = package.parent().path.decode('utf-8')
    sys.path.insert(0, pathEntry)
    self.addCleanup(sys.path.remove, pathEntry)
    from twisted_private_helper import module
    del sys.modules['twisted_private_helper']
    del sys.modules[module.__name__]
    try:
        from importlib import invalidate_caches
    except ImportError:
        pass
    else:
        invalidate_caches()
    package.moveTo(package.sibling(b'twisted_renamed_helper'))
    from twisted_renamed_helper import module
    self.addCleanup(sys.modules.pop, 'twisted_renamed_helper')
    self.addCleanup(sys.modules.pop, module.__name__)
    module.foo()
    self.assertEqual(len(self.flushWarnings([module.foo])), 1)