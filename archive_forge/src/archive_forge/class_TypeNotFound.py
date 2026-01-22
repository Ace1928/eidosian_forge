import sys
from .version import __build__, __version__
class TypeNotFound(Exception):

    def __init__(self, name):
        Exception.__init__(self, "Type not found: '%s'" % (tostr(name),))