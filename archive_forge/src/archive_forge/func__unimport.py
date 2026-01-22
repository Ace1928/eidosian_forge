import sys
from types import ModuleType
from twisted.trial.unittest import TestCase
def _unimport(self, name):
    """
        Find the given module and all of its hierarchically inferior modules in
        C{sys.modules}, remove them from it, and return whatever was found.
        """
    modules = {moduleName: module for moduleName, module in list(sys.modules.items()) if moduleName == self.name or moduleName.startswith(self.name + '.')}
    for name in modules:
        del sys.modules[name]
    return modules