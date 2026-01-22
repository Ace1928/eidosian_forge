from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import fnmatch
import glob
import importlib.util
import os
import pkgutil
import sys
import types
from googlecloudsdk.core.util import files
@contextlib.contextmanager
def GetFileTextReaderByLine(path):
    """Get a file reader for given path."""
    if os.path.isfile(path):
        f = files.FileReader(path)
        try:
            yield f
        finally:
            f.close()
    else:
        byte_string = GetResourceFromFile(path)
        yield str(byte_string, 'utf-8').split(os.linesep)