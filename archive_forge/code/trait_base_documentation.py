import enum
import os
import sys
from os import getcwd
from os.path import dirname, exists, join
from weakref import ref
from .etsconfig.api import ETSConfig
 Sets the value of an extended object attribute name of the form:
        name[.name2[.name3...]].
    