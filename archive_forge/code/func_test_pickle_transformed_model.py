from io import StringIO
import logging
from os.path import join, normpath
import pickle
from pyomo.common.fileutils import import_file, PYOMO_ROOT_DIR
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import (
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.tests.common_tests import (
from pyomo.gdp.tests.models import make_indexed_equality_model
from pyomo.repn import generate_standard_repn
def test_pickle_transformed_model(self):
    m = self.make_model()
    TransformationFactory('gdp.mbigm').apply_to(m, bigM=self.get_Ms(m))
    unpickle = pickle.loads(pickle.dumps(m))
    check_pprint_equal(self, m, unpickle)