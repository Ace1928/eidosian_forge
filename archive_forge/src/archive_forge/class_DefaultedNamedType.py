import sys
from pyasn1 import error
from pyasn1.type import tag
from pyasn1.type import tagmap
class DefaultedNamedType(NamedType):
    __doc__ = NamedType.__doc__
    isDefaulted = True