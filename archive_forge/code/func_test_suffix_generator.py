import collections.abc
import pickle
import pyomo.common.unittest as unittest
from pyomo.kernel import pprint
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.suffix import (
from pyomo.core.kernel.variable import variable
from pyomo.core.kernel.constraint import constraint, constraint_list
from pyomo.core.kernel.block import block, block_dict
def test_suffix_generator(self):
    m = block()
    m.s0 = suffix(direction=suffix.LOCAL)
    m.s0i = suffix(direction=suffix.LOCAL, datatype=suffix.INT)
    m.s1 = suffix(direction=suffix.IMPORT_EXPORT)
    m.s1i = suffix(direction=suffix.IMPORT_EXPORT, datatype=suffix.INT)
    m.s2 = suffix(direction=suffix.IMPORT)
    m.s2i = suffix(direction=suffix.IMPORT, datatype=suffix.INT)
    m.s3 = suffix(direction=suffix.EXPORT)
    m.s3i = suffix(direction=suffix.EXPORT, datatype=suffix.INT)
    m.b = block()
    m.b.s0 = suffix(direction=suffix.LOCAL)
    m.b.s0i = suffix(direction=suffix.LOCAL, datatype=suffix.INT)
    m.b.s1 = suffix(direction=suffix.IMPORT_EXPORT)
    m.b.s1i = suffix(direction=suffix.IMPORT_EXPORT, datatype=suffix.INT)
    m.b.s2 = suffix(direction=suffix.IMPORT)
    m.b.s2i = suffix(direction=suffix.IMPORT, datatype=suffix.INT)
    m.b.s3 = suffix(direction=suffix.EXPORT)
    m.b.s3i = suffix(direction=suffix.EXPORT, datatype=suffix.INT)
    self.assertEqual([id(c_) for c_ in suffix_generator(m)], [id(m.s0), id(m.s0i), id(m.s1), id(m.s1i), id(m.s2), id(m.s2i), id(m.s3), id(m.s3i), id(m.b.s0), id(m.b.s0i), id(m.b.s1), id(m.b.s1i), id(m.b.s2), id(m.b.s2i), id(m.b.s3), id(m.b.s3i)])
    self.assertEqual([id(c_) for c_ in suffix_generator(m, descend_into=False)], [id(m.s0), id(m.s0i), id(m.s1), id(m.s1i), id(m.s2), id(m.s2i), id(m.s3), id(m.s3i)])
    self.assertEqual([id(c_) for c_ in suffix_generator(m, datatype=suffix.INT)], [id(m.s0i), id(m.s1i), id(m.s2i), id(m.s3i), id(m.b.s0i), id(m.b.s1i), id(m.b.s2i), id(m.b.s3i)])
    m.s1.deactivate()
    m.b.deactivate()
    self.assertEqual([id(c_) for c_ in suffix_generator(m, active=True)], [id(m.s0), id(m.s0i), id(m.s1i), id(m.s2), id(m.s2i), id(m.s3), id(m.s3i)])