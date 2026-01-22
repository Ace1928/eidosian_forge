import functools
import inspect
import logging
import traceback
import wsme.exc
import wsme.types
from wsme import utils
class FunctionArgument(object):
    """
    An argument definition of an api entry
    """

    def __init__(self, name, datatype, mandatory, default):
        self.name = name
        self.datatype = datatype
        self.mandatory = mandatory
        self.default = default

    def resolve_type(self, registry):
        self.datatype = registry.resolve_type(self.datatype)