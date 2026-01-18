import enum
import os
import sys
from os import getcwd
from os.path import dirname, exists, join
from weakref import ref
from .etsconfig.api import ETSConfig
def xsetattr(object, xname, value):
    """ Sets the value of an extended object attribute name of the form:
        name[.name2[.name3...]].
    """
    names = xname.split('.')
    for name in names[:-1]:
        object = getattr(object, name)
    setattr(object, names[-1], value)