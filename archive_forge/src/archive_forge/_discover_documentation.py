import collections
import inspect
from automat import MethodicalMachine
from twisted.python.modules import PythonModule, getModule

    Recursively yield L{MethodicalMachine}s and their FQPNs in and
    under the a Python object specified by an FQPN.

    The discovery heuristic considers L{MethodicalMachine} instances
    that are module-level attributes or class-level attributes
    accessible from module scope.  Machines inside nested classes will
    be discovered, but those returned from functions or methods will not be.

    @type within: an FQPN
    @param within: Where to start the search.

    @return: a generator which yields FQPN, L{MethodicalMachine} pairs.
    