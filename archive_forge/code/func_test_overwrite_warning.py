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
def test_overwrite_warning(self):
    c = self._container_type()
    out = StringIO()
    with LoggingIntercept(out, 'pyomo.core'):
        c[0] = self._ctype_factory()
        c[0] = c[0]
    assert out.getvalue() == ''
    with LoggingIntercept(out, 'pyomo.core'):
        c[0] = self._ctype_factory()
    assert out.getvalue() == 'Implicitly replacing the entry [0] (type=%s) with a new object (type=%s). This is usually indicative of a modeling error. To avoid this warning, delete the original object from the container before assigning a new object.\n' % (self._ctype_factory().__class__.__name__, self._ctype_factory().__class__.__name__)