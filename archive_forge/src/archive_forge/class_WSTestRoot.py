import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
class WSTestRoot(WSRoot):
    argtypes = ArgTypes()
    returntypes = ReturnTypes()
    bodytypes = BodyTypes()
    witherrors = WithErrors()
    nested = NestedOuterApi()
    misc = MiscFunctions()

    def reset(self):
        self._touched = False

    @expose()
    def touch(self):
        self._touched = True