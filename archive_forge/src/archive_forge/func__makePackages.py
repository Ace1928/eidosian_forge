import sys
from types import ModuleType
from twisted.trial.unittest import TestCase
def _makePackages(parent, attributes, result):
    """
    Construct module objects (for either modules or packages).

    @param parent: L{None} or a module object which is the Python package
        containing all of the modules being created by this function call.  Its
        name will be prepended to the name of all created modules.

    @param attributes: A mapping giving the attributes of the particular module
        object this call is creating.

    @param result: A mapping which is populated with all created module names.
        This is suitable for use in updating C{sys.modules}.

    @return: A mapping of all of the attributes created by this call.  This is
        suitable for populating the dictionary of C{parent}.

    @see: L{_install}.
    """
    attrs = {}
    for name, value in list(attributes.items()):
        if parent is None:
            if isinstance(value, dict):
                module = ModuleType(name)
                module.__dict__.update(_makePackages(module, value, result))
                result[name] = module
            else:
                result[name] = value
        elif isinstance(value, dict):
            module = ModuleType(parent.__name__ + '.' + name)
            module.__dict__.update(_makePackages(module, value, result))
            result[parent.__name__ + '.' + name] = module
            attrs[name] = module
        else:
            attrs[name] = value
    return attrs