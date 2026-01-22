import pickle
import re
import sys
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def Importer(self, module, globals='', locals='', fromlist=None):
    """Importer function.

        Acts like __import__. Only loads modules from self.modules.
        Does not try to load real modules defined elsewhere. Does not
        try to handle relative imports.

        Args:
          module: Fully qualified name of module to load from self.modules.

        """
    if fromlist is None:
        module = module.split('.')[0]
    try:
        return self.modules[module]
    except KeyError:
        raise ImportError()