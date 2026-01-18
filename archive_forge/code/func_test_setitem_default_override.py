import math
import os
import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.errors import PyomoException
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base.param import _ParamData
from pyomo.core.base.set import _SetData
from pyomo.core.base.units_container import units, pint_available, UnitsError
from io import StringIO
def test_setitem_default_override(self):
    sparse_keys = set(self.instance.A.sparse_keys())
    keys = sorted(self.instance.A.keys())
    if len(keys) == len(sparse_keys):
        return
    if self.instance.A._default_val is NoValue:
        return
    while True:
        idx = keys.pop(0)
        if not idx in sparse_keys:
            break
    self.assertEqual(value(self.instance.A[idx]), self.instance.A._default_val)
    if self.instance.A.mutable:
        self.assertIsInstance(self.instance.A[idx], _ParamData)
    else:
        self.assertEqual(type(self.instance.A[idx]), type(value(self.instance.A._default_val)))
    try:
        self.instance.A[idx] = 4.3
        if not self.instance.A.mutable:
            self.fail('Expected setitem[%s] to fail for immutable Params' % (idx,))
        self.assertEqual(self.instance.A[idx].value, 4.3)
        self.assertIsInstance(self.instance.A[idx], _ParamData)
    except TypeError:
        if self.instance.A.mutable:
            raise
    try:
        self.instance.A[idx] = -4.3
        if not self.instance.A.mutable:
            self.fail('Expected setitem[%s] to fail for immutable Params' % (idx,))
        if self.expectNegativeDomainError:
            self.fail('Expected setitem[%s] to fail with negative data' % (idx,))
        self.assertEqual(self.instance.A[idx].value, -4.3)
    except ValueError:
        if not self.expectNegativeDomainError:
            self.fail('Unexpected exception (%s) for setitem[%s] = negative data' % (str(sys.exc_info()[1]), idx))
    except TypeError:
        if self.instance.A.mutable:
            raise
    try:
        self.instance.A[idx] = 'x'
        if not self.instance.A.mutable:
            self.fail('Expected setitem[%s] to fail for immutable Params' % (idx,))
        if self.expectTextDomainError:
            self.fail('Expected setitem[%s] to fail with text data' % (idx,))
        self.assertEqual(value(self.instance.A[idx]), 'x')
    except ValueError:
        if not self.expectTextDomainError:
            self.fail('Unexpected exception (%s) for setitem[%s] with text data' % (str(sys.exc_info()[1]), idx))
    except TypeError:
        if self.instance.A.mutable:
            raise