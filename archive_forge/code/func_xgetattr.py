import enum
import os
import sys
from os import getcwd
from os.path import dirname, exists, join
from weakref import ref
from .etsconfig.api import ETSConfig
def xgetattr(object, xname, default=Undefined):
    """ Returns the value of an extended object attribute name of the form:
        name[.name2[.name3...]].
    """
    names = xname.split('.')
    for name in names[:-1]:
        if default is Undefined:
            object = getattr(object, name)
        else:
            object = getattr(object, name, None)
            if object is None:
                return default
    if default is Undefined:
        return getattr(object, names[-1])
    return getattr(object, names[-1], default)