import os
import sys
from importlib import invalidate_caches as invalidateImportCaches
from twisted.trial import unittest
from twisted.trial import unittest
import unittest as pyunit
from twisted.trial import unittest
from twisted.trial import unittest
def _createDirectory(self, filename):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)