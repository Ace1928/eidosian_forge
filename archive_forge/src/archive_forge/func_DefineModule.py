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
def DefineModule(self, name):
    """Define a module and its parents in module space.

        Modules that are already defined in self.modules are not re-created.

        Args:
          name: Fully qualified name of modules to create.

        Returns:
          Deepest nested module.  For example:

            DefineModule('a.b.c')  # Returns c.
        """
    name_path = name.split('.')
    full_path = []
    for node in name_path:
        full_path.append(node)
        full_name = '.'.join(full_path)
        self.modules.setdefault(full_name, types.ModuleType(full_name))
    return self.modules[name]