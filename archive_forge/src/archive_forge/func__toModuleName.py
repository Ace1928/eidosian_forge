import os
import sys
from importlib import invalidate_caches as invalidateImportCaches
from twisted.trial import unittest
from twisted.trial import unittest
import unittest as pyunit
from twisted.trial import unittest
from twisted.trial import unittest
def _toModuleName(self, filename):
    name = os.path.splitext(filename)[0]
    segs = name.split('/')
    if segs[-1] == '__init__':
        segs = segs[:-1]
    return '.'.join(segs)