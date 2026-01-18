import copy
import itertools
import logging
import pickle
from io import StringIO
from collections import namedtuple as NamedTuple
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import pandas as pd, pandas_available
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr import native_numeric_types, native_types
import pyomo.core.base.set as SetModule
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.initializer import (
from pyomo.core.base.set import (
from pyomo.environ import (
def test_setproduct_construct_data(self):
    m = AbstractModel()
    m.I = Set(initialize=[1, 2])
    m.J = m.I * m.I
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):
        m.create_instance(data={None: {'J': {None: [(1, 1), (1, 2), (2, 1), (2, 2)]}}})
    self.assertRegex(output.getvalue().replace('\n', ' '), '^DEPRECATED: Providing construction data to SetOperator objects is deprecated')
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):
        with self.assertRaisesRegex(ValueError, 'Constructing SetOperator J with incompatible data \\(data=\\{None: \\[\\(1, 1\\), \\(1, 2\\), \\(2, 1\\)\\]\\}'):
            m.create_instance(data={None: {'J': {None: [(1, 1), (1, 2), (2, 1)]}}})
    self.assertRegex(output.getvalue().replace('\n', ' '), '^DEPRECATED: Providing construction data to SetOperator objects is deprecated')
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.core'):
        with self.assertRaisesRegex(ValueError, 'Constructing SetOperator J with incompatible data \\(data=\\{None: \\[\\(1, 3\\), \\(1, 2\\), \\(2, 1\\), \\(2, 2\\)\\]\\}'):
            m.create_instance(data={None: {'J': {None: [(1, 3), (1, 2), (2, 1), (2, 2)]}}})
    self.assertRegex(output.getvalue().replace('\n', ' '), '^DEPRECATED: Providing construction data to SetOperator objects is deprecated')