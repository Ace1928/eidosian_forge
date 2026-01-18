import copy
import copyreg as copy_reg
import inspect
import pickle
import types
from io import StringIO as _cStringIO
from typing import Dict
from twisted.python import log, reflect
from twisted.python.compat import _PYPY
def versionUpgrade(self):
    """(internal) Do a version upgrade."""
    bases = _aybabtu(self.__class__)
    bases.reverse()
    bases.append(self.__class__)
    if 'persistenceVersion' in self.__dict__:
        pver = self.__dict__['persistenceVersion']
        del self.__dict__['persistenceVersion']
        highestVersion = 0
        highestBase = None
        for base in bases:
            if 'persistenceVersion' not in base.__dict__:
                continue
            if base.persistenceVersion > highestVersion:
                highestBase = base
                highestVersion = base.persistenceVersion
        if highestBase:
            self.__dict__['%s.persistenceVersion' % reflect.qual(highestBase)] = pver
    for base in bases:
        if Versioned not in base.__bases__ and 'persistenceVersion' not in base.__dict__:
            continue
        currentVers = base.persistenceVersion
        pverName = '%s.persistenceVersion' % reflect.qual(base)
        persistVers = self.__dict__.get(pverName) or 0
        if persistVers:
            del self.__dict__[pverName]
        assert persistVers <= currentVers, "Sorry, can't go backwards in time."
        while persistVers < currentVers:
            persistVers = persistVers + 1
            method = base.__dict__.get('upgradeToVersion%s' % persistVers, None)
            if method:
                log.msg('Upgrading %s (of %s @ %s) to version %s' % (reflect.qual(base), reflect.qual(self.__class__), id(self), persistVers))
                method(self)
            else:
                log.msg('Warning: cannot upgrade {} to version {}'.format(base, persistVers))