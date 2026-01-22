import getopt
import inspect
import itertools
from types import MethodType
from typing import Dict, List, Set
from twisted.python import reflect, usage, util
from twisted.python.compat import ioType
def addAdditionalOptions(self) -> None:
    """
        Add additional options to the optFlags and optParams lists.
        These will be defined by 'opt_foo' methods of the Options subclass
        @return: L{None}
        """
    methodsDict: Dict[str, MethodType] = {}
    reflect.accumulateMethods(self.options, methodsDict, 'opt_')
    methodToShort = {}
    for name in methodsDict.copy():
        if len(name) == 1:
            methodToShort[methodsDict[name]] = name
            del methodsDict[name]
    for methodName, methodObj in methodsDict.items():
        longname = methodName.replace('_', '-')
        if longname in self.allOptionsNameToDefinition:
            continue
        descr = self.getDescription(longname)
        short = None
        if methodObj in methodToShort:
            short = methodToShort[methodObj]
        reqArgs = methodObj.__func__.__code__.co_argcount
        if reqArgs == 2:
            self.optParams.append([longname, short, None, descr])
            self.paramNameToDefinition[longname] = [short, None, descr]
            self.allOptionsNameToDefinition[longname] = [short, None, descr]
        else:
            self.optFlags.append([longname, short, descr])
            self.flagNameToDefinition[longname] = [short, descr]
            self.allOptionsNameToDefinition[longname] = [short, None, descr]