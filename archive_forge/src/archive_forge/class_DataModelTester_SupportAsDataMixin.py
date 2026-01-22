from llvmlite import ir
from llvmlite import binding as ll
from numba.core import datamodel
import unittest
class DataModelTester_SupportAsDataMixin(DataModelTester, SupportAsDataMixin):
    pass