import collections.abc
import pickle
from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.core.kernel.base import ICategorizedObject, ICategorizedObjectContainer
from pyomo.core.kernel.homogeneous_container import IHomogeneousContainer
from pyomo.core.kernel.dict_container import DictContainer
from pyomo.core.kernel.block import block, block_dict
def test_wrong_type_init(self):
    index = ['a', 1, None, (1,), (1, 2)]
    with self.assertRaises(TypeError):
        c = self._container_type(((i, _bad_ctype()) for i in index))